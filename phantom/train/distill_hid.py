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


def student_model_config(mc, *, mask_wrist: bool = False):
    """The student's PhantomModelConfig: the teacher's config with student=True,
    plus mask_wrist when the student must not see the wrist F/T either.
    mask_wrist is an input-ablation switch inside the model (HHT zeroes the
    wrist window), so it travels with the checkpoint into eval and deploy."""
    from phantom.config.model import PhantomModelConfig
    if mc is None:
        return PhantomModelConfig(student=True, mask_wrist=True) if mask_wrist else None
    return dataclasses.replace(mc, student=True, mask_wrist=bool(mask_wrist or mc.mask_wrist))

def hid_weights(teacher_pred, cfg: HIDConfig) -> torch.Tensor:
    """w_tau = s_tau * c_tau (B, Tc): saliency from teacher event probs,
    confidence from teacher sigma."""
    p_evt = teacher_pred.event_logits_B_Tc_E.softmax(-1)
    p_none = p_evt[..., EVENT_IDX["none"]]
    s = 1.0 + cfg.saliency_kappa * (1.0 - p_none)
    sigma = teacher_pred.log_sigma_B_Tc_K.exp().mean(-1)
    c = torch.exp(-sigma / cfg.confidence_temp)
    return (s * c).detach()


def teacher_sample_layout(teacher_rf, t_pred):
    """The SequenceLayout `teacher_rf.sample()` produced `t_pred` under.

    `sample()` rebuilds a drop_video layout when `mc.drop_video_at_inference`
    is set, so slicing `t_pred.x_final_B_C_T_H_W` with `teacher_rf.layout`
    would silently mis-slice such a checkpoint. Verified against the tensor
    rather than trusted."""
    layout = teacher_rf.layout
    if teacher_rf.mc.drop_video_at_inference:
        from phantom.model.sequence import SequenceLayout
        layout = SequenceLayout.build(teacher_rf.bb, teacher_rf.mc, teacher_rf.hw,
                                      student=layout.student, drop_video=True)
    t_total = int(t_pred.x_final_B_C_T_H_W.shape[2])
    if layout.t_total != t_total:
        raise RuntimeError(
            f"teacher imagination has T={t_total} frames but the layout "
            f"reconstructed for it has T={layout.t_total} — refusing to slice "
            f"the CONTACT frames out of a sequence whose layout is unknown")
    return layout


def traj_distill_target(student_rf, teacher_rf, t_pred, cfg: HIDConfig,
                        dtype) -> torch.Tensor:
    """The CONTACT-frame tensor traj_distill regresses the student onto.

    "roundtrip" (default, and what EVERY shipped checkpoint was distilled
    against) re-packs the teacher's UNPACKED contact package. That round trip
    is lossy in one way that matters: `ContactPacker.unpack` always returns a
    finite CoP centroid, and `pack` keys the CoP bump amplitude on NaN
    (packing.py `amp = (~isnan(cop))`), so every step where the teacher
    imagined NO contact comes back as a full-amplitude Gaussian bump at an
    arbitrary point — the exact fiction windows.py removed from the GT side on
    2026-08-26. It also low-passes d_disp/d_fz through cpk_shape and squashes
    the saturated event band to ~+-0.6.

    "raw_latents" distils against the teacher's raw CONTACT latents, which is
    the space `x0_pred_s` already lives in: no CoP fiction, no event squash, no
    resample. Opt-in, because it changes the objective and therefore the
    numbers (code defect, 2026-09-18 (a))."""
    mode = str(getattr(cfg, "traj_target", "roundtrip") or "roundtrip")
    if mode == "roundtrip":
        return student_rf.c_pack.pack(
            t_pred.cpk.detach().to(student_rf.device)).to(dtype)
    if mode == "raw_latents":
        sl_t = teacher_sample_layout(teacher_rf, t_pred).frame_slice(FrameGroup.CONTACT)
        return t_pred.x_final_B_C_T_H_W[:, :, sl_t].detach().to(
            student_rf.device, dtype)
    raise ValueError(f"traj_target={mode!r}: use 'roundtrip' or 'raw_latents'")


def distill_step(student_rf, teacher_rf, batch: dict, cfg: HIDConfig,
                 device: str) -> dict:
    batch = C.to_device(batch, device)
    layout_s = student_rf.layout

    # teacher imagination (no grad): the future the student must learn to feel
    # teacher NFE: round 1 sampled at nfe//2 (cost); -1 = the teacher's own
    # configured nfe (09-05: half-NFE targets are a distillation defect)
    tn = int(getattr(cfg, "teacher_nfe", 0) or 0)
    if tn < -1 or tn > 50:
        raise ValueError(f"teacher_nfe={tn}: use 0 (nfe//2), -1 (the teacher's nfe) or 1..50")
    nfe_t = (max(1, teacher_rf.mc.nfe // 2) if tn == 0
             else (int(teacher_rf.mc.nfe) if tn == -1 else tn))
    with torch.no_grad():
        t_pred = teacher_rf.sample(batch, nfe=nfe_t)
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

    # (i) trajectory distillation on the CONTACT frames. cfg.traj_target picks
    # the target space: "roundtrip" (default, shipped) re-packs the teacher's
    # unpacked package; "raw_latents" takes the teacher's CONTACT latents
    # directly. Same slice, same masking, same w_tau either way.
    cpk_target = traj_distill_target(student_rf, teacher_rf, t_pred, cfg, x0_s.dtype)
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
    # the student's video term rides on cfg.w_ground, NOT on mc.loss.video, so
    # only the ON/OFF of the weight is honoured here: lambda_v = 0 (inherited
    # from a --loss-video 0 teacher, or set by distill_hid's own --loss-video)
    # drops the term; any non-zero value keeps the shipped w_ground weighting.
    if (layout_s.has(FrameGroup.VIDEO_GEN)
            and float(student_rf.mc.loss.video) != 0.0):
        parts["video_v_mse"] = L.group_velocity_mse(v_s, v_target, layout_s,
                                                    FrameGroup.VIDEO_GEN)
    from phantom.model.ace.packing import _CH_WRIST
    parts["wrist_mse"] = L.wrist_region_mse(x0_pred_s, x0_s, layout_s, _CH_WRIST)
    if out_s.acc is not None:
        parts.update(L.acc_losses(out_s.acc, batch["gate_label"],
                                  batch["events"][:, 0]))

    # (v) student sigma head (09-05: round 1 left it at the teacher's
    # init with no gradient, so the deploy governor ran on an untrained head)
    # — the teacher's own heteroscedastic NLL vs the GT contact package
    if float(getattr(cfg, "w_sigma", 0.0) or 0.0) > 0:
        from phantom.model.rf import sigma_group_channels
        # a READOUT: the head calibrates on the student's own (detached)
        # hidden state and contact prediction; the trunk never sees this
        # loss — its contact frames belong to traj_distill (teacher
        # imagination), and an undetached NLL at beta 0.5 is ~10x an MSE
        # toward GT that would turn distillation back into supervised
        # training (verify 09-05 #1)
        log_sigma_s = student_rf.phantom_sigma_head(out_s.contact_hidden_B_Tc_S_D.detach())
        parts["contact_nll"] = L.contact_hetero_nll(
            x0_pred_s.detach(), x0_s, log_sigma_s, layout_s,
            group_channels=sigma_group_channels(student_rf.hw.n_fingers),
            beta=student_rf.mc.contact_nll_beta,
            detach_weight=student_rf.mc.contact_nll_detach_weight)
        parts["sigma_reg"] = (log_sigma_s ** 2).mean()

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
    if "contact_nll" in parts:
        # the teacher's own weighting (w.contact, w.sigma_reg), scaled by w_sigma
        total = (total + cfg.w_sigma * float(student_rf.mc.loss.contact) * parts["contact_nll"]
                 + cfg.w_sigma * float(student_rf.mc.loss.sigma_reg) * parts["sigma_reg"])
    parts["total"] = total
    return parts


def reconcile_teacher_ckpt(saved_train: dict, cfg: HIDConfig, args, *,
                           log=log) -> dict:
    """Let `--resume` accept the SAME teacher at a DIFFERENT path.

    The recipe lock compares `teacher_ckpt` as a raw string, so a student was
    permanently tied to the absolute workspace path of the box that trained it
    — `/workspace/phantom-hid/runs/teacher/...` — and could not be resumed on
    a rental that mounts the identical checkpoint elsewhere. From 2026-09-18
    the recipe also records `teacher_ckpt_sha12`; when the hash matches, the
    relocated path is accepted with a warning. A DIFFERENT hash is a different
    teacher and still refuses (2026-09-18 (d))."""
    saved_path = saved_train.get("teacher_ckpt")
    if saved_path is None or saved_path == cfg.teacher_ckpt:
        return saved_train
    if not cfg.teacher_ckpt:
        raise SystemExit(
            "--resume without --teacher-ckpt: the teacher is what supplies the "
            "model config and the imagination target, so it must be named "
            f"(the checkpoint was distilled from {saved_path!r})")
    saved_sha = str(saved_train.get("teacher_ckpt_sha12") or "")
    if not saved_sha:
        return saved_train          # checkpoint predates the hash: path rules
    if saved_sha != cfg.teacher_ckpt_sha12:
        return saved_train          # a DIFFERENT teacher: let the lock refuse
    log.warning("--resume: teacher checkpoint RELOCATED — recipe recorded %r, "
                "this run uses %r; sha256[:12] %s matches, accepting",
                saved_path, cfg.teacher_ckpt, saved_sha)
    saved = dict(saved_train)
    saved["teacher_ckpt"] = cfg.teacher_ckpt
    return saved


def eps_stack(eps_by_group: dict, layout) -> torch.Tensor:
    parts = [eps_by_group[s.group] for s in layout.slots]
    return torch.cat(parts, dim=2)


#: Teacher model-config flags that change how `training_step` noises a batch
#: but which `prepare_denoise` — the entry point BOTH sides of the
#: distillation go through — does not implement. Distilling a teacher that
#: sets one of them is not "slightly off": the teacher is evaluated under a
#: noising scheme it was never trained for, and nothing raises.
UNSUPPORTED_TEACHER_DENOISE_FLAGS = ("action_t_max_of_two", "contact_self_forcing")


def refuse_unsupported_teacher_flags(mc, *, allow: bool = False, log=log) -> None:
    """Refuse a teacher whose recorded config sets a denoise flag this
    program cannot honour (code-defect audit 2026-09-18, "Not addressed here").

    `RF.training_step` applies `action_t_max_of_two` (ACTION frames noised at
    max of two timestep draws) and `contact_self_forcing` (the CONTACT frames
    the ACTION frames are denoised alongside are the model's OWN two-pass
    prediction, not the co-noised GT). `RF.prepare_denoise`, which distill_hid
    uses for the student AND for the teacher's shared-noise branch, implements
    neither: it builds one `t_B_T` from the single draw and always noises the
    GT `x0`. So a teacher trained with either flag is queried off-distribution
    and its `behavior_match` target is not the velocity field it learned —
    silently, since the flags are recorded in the checkpoint and simply read
    back as inert.

    Dormant for the paper: all three checked-in `saved_model_config` dumps of
    `runs/teacher_v6/teacher_020000.pt` (sha256 4812cfbf0127...) record
    BOTH flags as False, matching the shipped launch recipe
    (`tools/provision_v5.sh` / `tools/provision_distill_v6.sh` pass
    `--no-action-t-max-of-two`; `--contact-self-forcing` was dropped from the
    FT-A bundle on 2026-08-30). So this never fires on a v6 distillation. It
    fires on the NEXT teacher that turns one on, which is the point — note
    that `train_teacher`'s CLI DEFAULT for `action_t_max_of_two` is True, so a
    teacher trained without `--no-action-t-max-of-two` will trip it.

    `--allow-unsupported-teacher-flags` downgrades the refusal to a warning for
    whoever wants the old (silent) behaviour deliberately."""
    if mc is None:
        return
    on = [f for f in UNSUPPORTED_TEACHER_DENOISE_FLAGS if bool(getattr(mc, f, False))]
    if not on:
        return
    msg = (f"teacher checkpoint sets {', '.join(on)}=True, and "
           f"distill_hid cannot honour it: `RF.prepare_denoise` — the noising "
           f"path BOTH the student and the teacher's shared-noise branch go "
           f"through — implements neither `action_t_max_of_two` (per-group "
           f"ACTION timestep = max of two draws) nor `contact_self_forcing` "
           f"(ACTION frames denoised alongside the model's own predicted "
           f"contact package), while `RF.training_step` applies both. "
           f"Distilling this teacher would query it off its training "
           f"distribution and silently mis-derive the behavior_match target. "
           f"Retrain the teacher without the flag, teach prepare_denoise to "
           f"honour it, or pass --allow-unsupported-teacher-flags to proceed "
           f"anyway (the pre-2026-09-18 behaviour, silent until now).")
    if allow:
        log.warning("--allow-unsupported-teacher-flags: %s", msg)
        return
    raise ValueError(msg)


# ---------------------------------------------------------------------------

def build_parser(ap: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """distill_hid's CLI. Split out of `main` so a test can go
    argparse -> HIDConfig -> effect without launching a run."""
    ap = ap or argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--teacher-ckpt", default="")
    ap.add_argument("--ckpt-every", type=int, default=None,
                    help="checkpoint cadence (default 1000; clipped to --max-steps)")
    ap.add_argument("--eval-every", type=int, default=None,
                    help="val cadence (default 1000; clipped to --max-steps)")
    ap.add_argument("--dagger-round", type=int, default=0)
    ap.add_argument("--mask-wrist", action="store_true",
                    help="student INPUT ablation: zero the wrist F/T window for the student "
                         "(camera + proprio only = genuinely sensor-free; the teacher keeps "
                         "pads + wrist). The flag is stored in the student checkpoint's model "
                         "config, so terminal_eval / run_deploy zero it again automatically.")
    ap.add_argument("--loss-video", type=float, default=None,
                    help="override the STUDENT's lambda_v (default: inherited "
                         "from the teacher checkpoint). Only 0 vs non-zero "
                         "matters here — the student's video term is weighted "
                         "by w_ground, so 0 drops it and any other value keeps "
                         "the shipped weighting. Stored in the student's model "
                         "config.")
    ap.add_argument("--extra-data", nargs="*", default=[],
                    help="additional episode roots (DAgger rollouts)")
    ap.add_argument("--resume", default="",
                    help="student checkpoint to resume from (restores weights, "
                         "optimizer, schedule, EMA and step — the runner's "
                         "incremental hub egress makes a preempted rental "
                         "resumable; 2026-09-05)")
    # the teachers were trained with --grasp-frac 0.3 --photo-aug 1.0; the
    # student must see the same window recipe (2026-09-05: uniform t0
    # gave the pre-close commit band ~5-10% of windows instead of 30%+ and
    # collection lighting only).
    # All four default to None so `--resume` can tell "the operator chose the
    # default" from "nobody said anything"; the dataclass carries the real
    # default (0.0 / 0.0 / 1.0 / train), unchanged (2026-09-18 (c)).
    ap.add_argument("--grasp-frac", type=float, default=None,
                    help="fraction of windows anchored in the 1.5 s before the "
                         "first gripper close (default 0 = uniform). Persisted "
                         "in the student checkpoint and restored on --resume.")
    ap.add_argument("--photo-aug", type=float, default=None,
                    help="scene-camera photometric jitter strength (default 0 = "
                         "off). Persisted in the checkpoint and restored on "
                         "--resume.")
    ap.add_argument("--commit-band-weight", type=float, default=None,
                    help="ACTION-loss multiplier for windows in the pre-close "
                         "commit band (default 1.0 = off). Persisted in the "
                         "checkpoint and restored on --resume.")
    ap.add_argument("--traj-target", default=None,
                    choices=["roundtrip", "raw_latents"],
                    help="target space of traj_distill. roundtrip (DEFAULT, and "
                         "what every shipped v5/v6 student was distilled "
                         "against): pack(unpack(teacher package)) — a lossy "
                         "round trip that paints a full-amplitude CoP bump on "
                         "every step the teacher imagined as NO-contact, "
                         "because the packer keys the bump amplitude on NaN and "
                         "unpack never returns NaN. raw_latents: the teacher's "
                         "raw CONTACT latents (detached), the space the "
                         "student's x0 already lives in — no CoP fiction, no "
                         "event-band squash, no cpk_shape resample. Changes the "
                         "objective, so it is NOT resume-compatible with a "
                         "roundtrip student. Recorded in the checkpoint.")
    ap.add_argument("--rollout-action-weight", default=None,
                    choices=["failure_demo", "judged_rollouts"],
                    help="action grounding on on-policy (DAgger) rollouts. "
                         "failure_demo (DEFAULT, shipped): `is_failure_demo` "
                         "fires on a `success is False` verdict, so a rollout "
                         "the operator judged failed gets action_weight 0 — in "
                         "round 2 that was ~92%% of the 99 rollouts, i.e. the "
                         "overlay grounded no actions at all. "
                         "judged_rollouts: a POLICY rollout marked failed "
                         "ONLY by the verdict keeps its per-episode weight "
                         "(deliberate failure demos and rollouts whose actions "
                         "are still the executor proposal stay at 0). The mode "
                         "is named for the EPISODES it re-admits, not for a "
                         "supervisor: what it grounds is the rollout's OWN "
                         "re-derived (measured delta-EE) action stream on the "
                         "rollouts a verdict judged — it is NOT teacher "
                         "relabelling, and no teacher action ever enters the "
                         "GT term. "
                         "behavior_match and traj_distill ignore action_weight "
                         "and are unaffected either way. Recorded in the "
                         "checkpoint.")
    ap.add_argument("--allow-unsupported-teacher-flags", action="store_true",
                    help="proceed even when the teacher's recorded config sets "
                         "action_t_max_of_two or contact_self_forcing, which "
                         "RF.training_step applies but RF.prepare_denoise (the "
                         "noising path distillation uses for BOTH models) does "
                         "not — so the teacher is queried off its training "
                         "distribution and behavior_match's target is not the "
                         "velocity field it learned. Off, the run refuses; on, "
                         "it warns. No v5/v6 teacher sets either flag.")
    ap.add_argument("--teacher-nfe", type=int, default=None,
                    help="teacher imagination NFE: 0 = nfe//2 (round 1), -1 = the "
                         "teacher's own nfe, N = N")
    ap.add_argument("--w-sigma", type=float, default=None,
                    help="weight of the student sigma-head NLL vs GT (0 = head "
                         "untrained, round 1)")
    # haptic-imagination ablation (09-13): the three distillation terms that carry the
    # teacher's IMAGINED contact into the student (traj_distill on the contact package,
    # event_distill on the event band) vs plain action matching (behavior_match).
    # 0 drops a term; None keeps the shipped weight (HIDConfig).
    ap.add_argument("--w-traj", type=float, default=None, help="traj_distill weight (contact package); 0 = off")
    ap.add_argument("--w-event", type=float, default=None, help="event_distill weight (event band); 0 = off")
    ap.add_argument("--w-behavior", type=float, default=None, help="behavior_match weight (action velocity matching); 0 = off")
    ap.add_argument("--split", default=None, choices=["train", "val", "all"],
                    help="episode subset from manifests/all.jsonl (default "
                         "train; 'all' reproduces the pre-2026-08-30 behaviour "
                         "of training on the held-out val episodes too). "
                         "Persisted in the checkpoint and restored on --resume.")
    return ap


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)

    rank, world = C.setup_ddp()   # EARLY: "cuda" resolves per-rank from here on
    profile = load_compute(args.compute)
    profile.check_world(world)
    comp = profile.for_program("distill_hid")

    hw = load_hardware(args.hardware)
    paths = load_paths()
    paths.validate(require_cosmos=not args.tiny)
    cfg = apply_overrides(HIDConfig(), args, compute=comp)
    # the teacher is recorded by PATH *and* by content hash: the path is what
    # the operator typed, the hash is what a --resume can match after the
    # rental workspace moved (2026-09-18 (d))
    teacher_sha = (C.file_sha12(args.teacher_ckpt) if args.teacher_ckpt
                   and Path(args.teacher_ckpt).exists() else "")
    cfg = dataclasses.replace(cfg, teacher_ckpt=args.teacher_ckpt,
                              dagger_round=args.dagger_round,
                              teacher_ckpt_sha12=teacher_sha)
    out_dir = paths.runs_root / "hid" / f"{cfg.run_name}_r{cfg.dagger_round}"
    dtype = C.pick_dtype(args.device, args.tiny, comp)

    # teacher (frozen) + student (trainable), student initialized from teacher.
    # Both are built with the TEACHER CHECKPOINT'S OWN model config (P10B,
    # 2026-08-28): building code defaults gave the teacher
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
    # BEFORE anything is built or loaded: a teacher whose noising scheme this
    # program cannot reproduce is refused here, not discovered in the numbers
    # (code-defect audit 2026-09-18, "Not addressed here")
    refuse_unsupported_teacher_flags(
        mc, allow=bool(getattr(args, "allow_unsupported_teacher_flags", False)))
    mc_teacher = dataclasses.replace(mc, student=False) if mc else None
    mc_student = student_model_config(mc, mask_wrist=bool(getattr(args, "mask_wrist", False)))
    if args.loss_video is not None:
        if args.loss_video < 0:
            raise SystemExit(f"--loss-video must be >= 0, got {args.loss_video}")
        from phantom.config.model import LossWeights
        base = mc_student.loss if mc_student is not None else LossWeights()
        loss = dataclasses.replace(base, video=float(args.loss_video))
        mc_student = (dataclasses.replace(mc_student, loss=loss)
                      if mc_student is not None
                      else PhantomModelConfig(student=True, loss=loss))
        log.info("STUDENT lambda_v overridden: %.4g%s", loss.video,
                 " — video grounding term DROPPED" if loss.video == 0 else "")
    teacher = build_model(hw, paths, student=False, tiny=cfg.tiny, mc=mc_teacher,
                          load_base=not cfg.tiny, device=args.device, dtype=dtype)
    student = build_model(hw, paths, student=True, tiny=cfg.tiny, mc=mc_student,
                          load_base=not cfg.tiny, device=args.device, dtype=dtype)
    if cfg.teacher_ckpt:
        # EMA is the DEPLOY/EVAL artifact (run_deploy --ema default, replay_rig,
        # terminal_eval): the teachers were SELECTED on their EMA weights, so
        # the imagination target and the student's starting point must be
        # those weights, not the raw last step (2026-09-05 must-fix).
        has_ema = bool((payload or {}).get("ema"))
        log.info("teacher weights: %s", "EMA" if has_ema else
                 "RAW (checkpoint carries no EMA)")
        C.load_phantom_checkpoint(Path(cfg.teacher_ckpt), teacher.rf, hw=hw,
                                  payload=payload, load_ema=has_ema)
        C.load_phantom_checkpoint(Path(cfg.teacher_ckpt), student.rf, hw=hw,
                                  allow_missing=True,   # teacher-only keys dropped
                                  payload=payload, load_ema=has_ema,
                                  # an explicit --loss-video is a deliberate
                                  # objective change vs the teacher's config
                                  tolerate_model_fields=(
                                      frozenset({"loss.video"})
                                      if args.loss_video is not None
                                      else frozenset()))
    teacher.rf.eval()
    for p in teacher.rf.parameters():
        p.requires_grad = False

    resume_payload = None
    if args.resume:
        resume_payload = C.load_phantom_checkpoint(Path(args.resume), student.rf, hw=hw)
        # a resume continues ONE run: every recipe key (w_sigma, teacher_nfe,
        # grasp_frac, ...) comes back from the checkpoint unless the CLI names
        # it, and a DIFFERENT CLI value refuses the resume — the teacher's
        # rule (2026-08-31 #8), which this program skipped
        # (verify 09-05 #2)
        from phantom.train.train_teacher import restore_train_config_on_resume
        saved_train = reconcile_teacher_ckpt(
            resume_payload["configs"].get("train") or {}, cfg, args)
        cfg = restore_train_config_on_resume(cfg, saved_train, args)
        # the hash of the teacher this segment ACTUALLY loaded, not the
        # checkpoint's record of an earlier path
        cfg = dataclasses.replace(cfg, teacher_ckpt_sha12=teacher_sha)
        log.info("resuming student from %s at step %d", args.resume,
                 int(resume_payload.get("step", 0)))

    data_root, norm = resolve_data(args, hw, paths)
    # inherit the teacher/student checkpoint's wrench zero-offset rows and
    # PERSIST them in this program's checkpoint (verify 09-05 #1)
    cfg = dataclasses.replace(cfg, wrench_baseline_rows=C.wrench_baseline_rows_of(payload))
    sampler_s = WindowSampler(hw, student.bb, norm, student=False, seed=cfg.seed,
                              wrench_baseline_rows=cfg.wrench_baseline_rows,
                              rollout_action_weight=cfg.rollout_action_weight)
    # note: student windows still carry tactile targets (labels come from the
    # rig's sensors during training); the student MODEL just never sees the
    # tactile inputs — its layout has no OBS_GEL/OBS_MECH frames.
    # the manifest split, exactly as train_teacher does it: without
    # `episodes=` WindowDataset falls through to list_episodes(root) and the
    # student trains on its own validation set (2026-08-30 F10)
    # the window recipe comes off CFG, not off args: that is what a --resume
    # restores from the checkpoint (2026-09-18 (c))
    train_eps = C.manifest_split(data_root, cfg.split)
    ds = C.WindowDataset(data_root, sampler_s, episodes=train_eps,
                         grasp_frac=cfg.grasp_frac, photo_aug=cfg.photo_aug,
                         commit_band_weight=cfg.commit_band_weight)
    for extra in args.extra_data:
        ds.index += sampler_s.build_index(Path(extra))
    log.info("HID dataset: %d windows (round %d, split=%s, grasp_frac=%.2f, "
             "photo_aug=%.2f, traj_target=%s, rollout_action_weight=%s)",
             len(ds), cfg.dagger_round, cfg.split, cfg.grasp_frac,
             cfg.photo_aug, cfg.traj_target, cfg.rollout_action_weight)
    loader = C.make_loader(ds, cfg)   # AFTER the --extra-data index merge

    # held-out evaluation, exactly as train_teacher builds it: without a
    # val_loader train_loop's eval_every is dead and no checkpoint can be
    # selected on anything but the teacher-matching training loss
    val_loader = None
    if cfg.split == "train":
        val_eps = C.manifest_split(data_root, "val")
        if val_eps:
            val_ds = C.WindowDataset(data_root, sampler_s, episodes=val_eps,
                                     resample=False, seed=cfg.seed)
            val_loader = C.make_loader(val_ds, cfg, shuffle=False, shard=False)
            log.info("val: %d windows from %d episodes", len(val_ds), len(val_eps))
    if not cfg.synthetic and not cfg.tiny:
        C.assert_label_sanity(ds, log)

    def step_fn(batch: dict) -> dict:
        return distill_step(student.rf, teacher.rf, batch, cfg, args.device)

    def on_ckpt(step, opt, sched, ema):
        C.save_phantom_checkpoint(
            out_dir / f"student_{step:06d}.pt", student.rf, hw=hw, bb=student.bb,
            mc=student.mc, train_cfg=cfg, step=step,
            base_ckpt_path=str(paths.cosmos_checkpoint), norm_stats=norm,
            optimizer=opt, scheduler=sched, ema=ema)

    C.train_loop(cfg, student.rf, loader, step_fn, on_checkpoint=on_ckpt,
                 val_loader=val_loader, resume_payload=resume_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
