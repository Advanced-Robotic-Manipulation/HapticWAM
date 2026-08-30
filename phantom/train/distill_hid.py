"""Program (3): HID — Haptic-Imagination Distillation (pipeline.md §5).

Teacher: full PHANTOM (frozen, program-2 checkpoint). Student: same classes
with student=True layout (no DM-Tac inputs), initialized from the teacher's
shared weights. Losses:

  (i)  trajectory distillation — student CONTACT-frame x0 regressed to the
       teacher's IMAGINED future contact package, per-step weight
       w_tau = s_tau * c_tau (event saliency x teacher confidence);
  (ii) event CE against the teacher's soft event predictions;
  (iii) behavior matching — ACTION-frame velocity matching at shared (x_t, t)
       (identical noise on shared groups; diffusion-native pi-matching);
  (iv) GT grounding — real actions/video/wrist RF terms + ACC auxiliaries.

teacher_mode:
  "online" — teacher.sample() per batch (simple; costs teacher forwards);
  "cached" — precompute pass (this module's precompute_relabels; also used by
       the DAgger driver) writes teacher outputs per window index, then
       training reads them — preferred on 8xH100.

Launch:
    torchrun --nproc_per_node 8 -m phantom.train.distill_hid \
        --teacher-ckpt runs/teacher/.../teacher_050000.pt --data <episodes_root>
Smoke: python -m phantom.train.distill_hid --tiny --synthetic --max-steps 2
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from phantom.config.compute import load_compute
from phantom.config.hardware import load_hardware
from phantom.config.model import EVENT_IDX, PhantomModelConfig
from phantom.config.paths import load_paths
from phantom.config.training import HIDConfig
from phantom.data.windows import WindowSampler
from phantom.model.ace import losses as L
from phantom.model.sequence import FrameGroup
from phantom.train import common as C
from phantom.train.builder import build_model
from phantom.train.train_teacher import add_common_args, apply_overrides, resolve_data

log = logging.getLogger("distill_hid")


# ---------------------------------------------------------------------------

def hid_weights(teacher_pred, cfg: HIDConfig) -> torch.Tensor:
    """w_tau = s_tau * c_tau (B, Tc): saliency from teacher event probs,
    confidence from teacher sigma."""
    p_evt = teacher_pred.event_logits_B_Tc_E.softmax(-1)
    p_none = p_evt[..., EVENT_IDX["none"]]
    s = 1.0 + cfg.saliency_kappa * (1.0 - p_none)
    sigma = teacher_pred.log_sigma_B_Tc_K.exp().mean(-1)
    c = torch.exp(-sigma / cfg.confidence_temp)
    return (s * c).detach()


def distill_step(student_rf, teacher_rf, batch: dict, cfg: HIDConfig,
                 device: str) -> dict:
    batch = C.to_device(batch, device)
    layout_s = student_rf.layout

    # teacher imagination (no grad): the future the student must learn to feel
    with torch.no_grad():
        t_pred = teacher_rf.sample(batch, nfe=max(1, teacher_rf.mc.nfe // 2))
    w_tau = hid_weights(t_pred, cfg)                          # (B, Tc)

    # student forward at sampled t, teacher velocity at the SAME (t, shared eps)
    B = batch["video"].shape[0]
    t_B = student_rf._sample_t(B, student_rf.device)
    x0_s, x_t_s, cond_s, acc_s, tBT_s, eps_used = student_rf.prepare_denoise(batch, t_B)
    out_s = student_rf.velocity_at(batch, x_t_s, tBT_s, cond_s, acc_s)
    v_s = out_s.velocity_B_C_T_H_W
    t_full = t_B.reshape(B, 1, 1, 1, 1).to(x0_s.dtype)
    x0_pred_s = x_t_s - t_full * v_s

    parts: dict[str, torch.Tensor] = {}

    # (i) trajectory distillation on the CONTACT frames (teacher-packed target)
    cpk_target = student_rf.c_pack.pack(t_pred.cpk.detach().to(student_rf.device)) \
        .to(x0_s.dtype)
    sl = layout_s.frame_slice(FrameGroup.CONTACT)
    d = (x0_pred_s[:, :, sl].float() - cpk_target.float()) ** 2
    parts["traj_distill"] = (d.mean(dim=(1, 3, 4)) * w_tau).mean()

    # (ii) event CE vs teacher soft predictions
    ev_s = student_rf.phantom_event_head(out_s.contact_hidden_B_Tc_S_D)
    t_probs = t_pred.event_logits_B_Tc_E.softmax(-1).detach()
    parts["event_distill"] = F.kl_div(ev_s.log_softmax(-1), t_probs,
                                      reduction="batchmean")

    # (iii) behavior matching: teacher velocity at shared t + shared eps on
    # the groups both layouts contain
    shared = {g: eps_used[g] for g in eps_used
              if teacher_rf.layout.has(g)}
    with torch.no_grad():
        _, x_t_t, cond_t, acc_t, tBT_t, _ = teacher_rf.prepare_denoise(
            batch, t_B, eps_by_group=shared)
        v_t = teacher_rf.velocity_at(batch, x_t_t, tBT_t, cond_t, acc_t) \
            .velocity_B_C_T_H_W
    sl_s = layout_s.frame_slice(FrameGroup.ACTION)
    sl_t = teacher_rf.layout.frame_slice(FrameGroup.ACTION)
    # live channels only — matching the packer's 12 zero-padding channels
    # would let padding noise dominate distillation (w_behavior=1.0)
    ch = slice(0, layout_s.actions_per_frame)
    parts["behavior_match"] = F.mse_loss(v_s[:, ch, sl_s].float(),
                                         v_t[:, ch, sl_t].detach().float())

    # (iv) GT grounding on actions / video / wrist + ACC auxiliaries
    v_target = eps_stack(eps_used, layout_s) - x0_s
    # same rule as the teacher: never ground actions on deliberate-failure
    # demos (their behavior is what we do NOT want the student to imitate)
    act_w = batch.get("action_weight")
    if act_w is not None:
        act_w = torch.as_tensor(act_w, device=v_s.device).reshape(-1)
    parts["action_v_mse"] = L.group_velocity_mse(
        v_s, v_target, layout_s, FrameGroup.ACTION, act_w,
        channels=slice(0, layout_s.actions_per_frame))
    if layout_s.has(FrameGroup.VIDEO_GEN):
        parts["video_v_mse"] = L.group_velocity_mse(v_s, v_target, layout_s,
                                                    FrameGroup.VIDEO_GEN)
    from phantom.model.ace.packing import _CH_WRIST
    parts["wrist_mse"] = L.wrist_region_mse(x0_pred_s, x0_s, layout_s, _CH_WRIST)
    if out_s.acc is not None:
        parts.update(L.acc_losses(out_s.acc, batch["gate_label"],
                                  batch["events"][:, 0]))

    # optional feature alignment (Efficient-WAM-style; single ablation row)
    if cfg.feature_align:
        with torch.no_grad():
            h_t = teacher_rf.velocity_at(batch, x_t_t, tBT_t, cond_t, acc_t) \
                .contact_hidden_B_Tc_S_D
        h_s = out_s.contact_hidden_B_Tc_S_D
        parts["feature_align"] = (1.0 - F.cosine_similarity(
            h_s.float().mean(2), h_t.float().mean(2).detach(), dim=-1)).mean()

    total = (cfg.w_traj * parts["traj_distill"]
             + cfg.w_event * parts["event_distill"]
             + cfg.w_behavior * parts["behavior_match"]
             + cfg.w_ground * (parts["action_v_mse"]
                               + parts.get("video_v_mse", torch.zeros((), device=device))
                               + parts["wrist_mse"]))
    if "acc_gate_bce" in parts:
        total = total + 0.2 * parts["acc_gate_bce"] + 0.5 * parts["acc_event_ce"]
    if cfg.feature_align:
        total = total + cfg.w_feature_align * parts["feature_align"]
    parts["total"] = total
    return parts


def eps_stack(eps_by_group: dict, layout) -> torch.Tensor:
    parts = [eps_by_group[s.group] for s in layout.slots]
    return torch.cat(parts, dim=2)


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--teacher-ckpt", default="")
    ap.add_argument("--dagger-round", type=int, default=0)
    ap.add_argument("--extra-data", nargs="*", default=[],
                    help="additional episode roots (DAgger rollouts)")
    ap.add_argument("--split", default="train", choices=["train", "val", "all"],
                    help="episode subset from manifests/all.jsonl (default "
                         "train; 'all' reproduces the pre-2026-08-30 behaviour "
                         "of training on the held-out val episodes too)")
    args = ap.parse_args(argv)

    rank, world = C.setup_ddp()   # EARLY: "cuda" resolves per-rank from here on
    profile = load_compute(args.compute)
    profile.check_world(world)
    comp = profile.for_program("distill_hid")

    hw = load_hardware(args.hardware)
    paths = load_paths()
    paths.validate(require_cosmos=not args.tiny)
    cfg = apply_overrides(HIDConfig(), args, compute=comp)
    cfg = dataclasses.replace(cfg, teacher_ckpt=args.teacher_ckpt,
                              dagger_round=args.dagger_round)
    out_dir = paths.runs_root / "hid" / f"{cfg.run_name}_r{cfg.dagger_round}"
    dtype = C.pick_dtype(args.device, args.tiny, comp)

    # teacher (frozen) + student (trainable), student initialized from teacher.
    # Both are built with the TEACHER CHECKPOINT'S OWN model config (P10B,
    # review 2026-08-28): building code defaults gave the teacher
    # rope_time_mode='aligned' + acc='gt_noised' while v4/v5 were trained
    # 'time_true' + 'two_pass', so the distillation target came from a
    # differently-phased model and nothing raised.
    mc = payload = None
    if cfg.teacher_ckpt:
        payload = torch.load(str(Path(cfg.teacher_ckpt)), map_location="cpu",
                             weights_only=False)
        saved_mc = (payload.get("configs") or {}).get("model")
        if isinstance(saved_mc, dict):
            mc = PhantomModelConfig.from_dict(saved_mc)
            log.info("model config from teacher checkpoint: rope=%s acc=%s "
                     "cond_dropout=%.2f", mc.rope_time_mode,
                     mc.acc.self_anticipation, mc.cond_dropout_p)
    mc_teacher = dataclasses.replace(mc, student=False) if mc else None
    mc_student = dataclasses.replace(mc, student=True) if mc else None
    teacher = build_model(hw, paths, student=False, tiny=cfg.tiny, mc=mc_teacher,
                          load_base=not cfg.tiny, device=args.device, dtype=dtype)
    student = build_model(hw, paths, student=True, tiny=cfg.tiny, mc=mc_student,
                          load_base=not cfg.tiny, device=args.device, dtype=dtype)
    if cfg.teacher_ckpt:
        C.load_phantom_checkpoint(Path(cfg.teacher_ckpt), teacher.rf, hw=hw,
                                  payload=payload)
        C.load_phantom_checkpoint(Path(cfg.teacher_ckpt), student.rf, hw=hw,
                                  allow_missing=True,   # teacher-only keys dropped
                                  payload=payload)
    teacher.rf.eval()
    for p in teacher.rf.parameters():
        p.requires_grad = False

    data_root, norm = resolve_data(args, hw, paths)
    sampler_s = WindowSampler(hw, student.bb, norm, student=False, seed=cfg.seed)
    # note: student windows still carry tactile targets (labels come from the
    # rig's sensors during training); the student MODEL just never sees the
    # tactile inputs — its layout has no OBS_GEL/OBS_MECH frames.
    # the manifest split, exactly as train_teacher does it: without
    # `episodes=` WindowDataset falls through to list_episodes(root) and the
    # student trains on its own validation set (validation 2026-08-30 F10)
    train_eps = C.manifest_split(data_root, args.split)
    ds = C.WindowDataset(data_root, sampler_s, episodes=train_eps)
    for extra in args.extra_data:
        ds.index += sampler_s.build_index(Path(extra))
    log.info("HID dataset: %d windows (round %d, split=%s)", len(ds),
             cfg.dagger_round, args.split)
    loader = C.make_loader(ds, cfg)   # AFTER the --extra-data index merge

    def step_fn(batch: dict) -> dict:
        return distill_step(student.rf, teacher.rf, batch, cfg, args.device)

    def on_ckpt(step, opt, sched, ema):
        C.save_phantom_checkpoint(
            out_dir / f"student_{step:06d}.pt", student.rf, hw=hw, bb=student.bb,
            mc=student.mc, train_cfg=cfg, step=step,
            base_ckpt_path=str(paths.cosmos_checkpoint), norm_stats=norm,
            optimizer=opt, scheduler=sched, ema=ema)

    C.train_loop(cfg, student.rf, loader, step_fn, on_checkpoint=on_ckpt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
