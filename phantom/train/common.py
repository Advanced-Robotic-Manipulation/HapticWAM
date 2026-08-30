"""Shared training infrastructure: DDP setup, param groups, EMA, checkpoint
format, dataset wrappers, and the generic training loop used by all four
programs. Plain PyTorch — no imaginaire trainer (user decision)."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

from phantom.config.backbone import BackboneConfig
from phantom.config.hardware import HardwareConfig
from phantom.config.model import TRAIN_ONLY_MODEL_FIELDS, PhantomModelConfig
from phantom.config.training import CommonTrainConfig
from phantom.data.schema import (NON_TRAINING_TAGS, EpisodeMeta, NormStats,
                                 is_trainable_episode)

log = logging.getLogger(__name__)

CKPT_FORMAT_VERSION = 1


# ---------------------------------------------------------------------------
# distributed
# ---------------------------------------------------------------------------

def setup_ddp() -> tuple[int, int]:
    """(rank, world_size); initializes the process group under torchrun.
    Idempotent — call it EARLY in a program's main() so a bare "cuda" resolves
    to this rank's device for model/loader construction (the VAE and the text
    provider are unregistered attrs the loop's model.to() never moves), and
    again (no-op) inside train_loop."""
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    if "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        backend = ("nccl" if torch.cuda.is_available() and dist.is_nccl_available()
                   else "gloo")
        dist.init_process_group(backend=backend)
        rank, world = dist.get_rank(), dist.get_world_size()
        if torch.cuda.is_available():
            local = int(os.environ.get("LOCAL_RANK",
                                       rank % torch.cuda.device_count()))
            torch.cuda.set_device(local)
        if rank != 0:
            logging.getLogger().setLevel(logging.WARNING)  # rank 0 keeps INFO
        return rank, world
    return 0, 1


def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def pick_dtype(device: str, tiny: bool, compute=None) -> torch.dtype:
    """--tiny is always fp32 (CPU smoke, even with a cluster --compute); else
    the compute profile's dtype if set; else the legacy rule (bf16 on cuda,
    fp32 otherwise) — bit-identical to the historical inline expression when
    no profile dtype is given."""
    if tiny:
        return torch.float32
    if compute is not None and compute.dtype is not None:
        return torch.bfloat16 if compute.dtype == "bf16" else torch.float32
    return torch.bfloat16 if device == "cuda" else torch.float32


# ---------------------------------------------------------------------------
# optimizer / EMA
# ---------------------------------------------------------------------------

class Fp32MasterAdamW(torch.optim.AdamW):
    """AdamW stepping fp32 MASTER copies of (bf16) trainable params.

    Pure-bf16 training silently froze every parameter whose per-step update
    fell below the bf16 ulp — measured on teacher v3: acc beta_raw and all
    tactile-encoder norm scales never moved off init (v4 audit 2026-08-14).
    The masters accumulate updates in fp32 and are copied back to the model
    dtype each step; grads flow from the live (bf16) params."""

    def __init__(self, param_groups, **kw):
        self._live: list[torch.nn.Parameter] = []
        master_groups = []
        for g in param_groups:
            live = list(g["params"])
            masters = [torch.nn.Parameter(p.detach().float().clone(),
                                          requires_grad=False) for p in live]
            self._live.extend(live)
            master_groups.append({**g, "params": masters})
        super().__init__(master_groups, **kw)
        self._masters = [p for g in self.param_groups for p in g["params"]]
        assert len(self._masters) == len(self._live)

    @torch.no_grad()
    def step(self, closure=None):
        for live, master in zip(self._live, self._masters):
            master.grad = live.grad.float() if live.grad is not None else None
        loss = super().step(closure)
        for live, master in zip(self._live, self._masters):
            live.data.copy_(master.data.to(live.dtype))
            master.grad = None
        return loss

    def zero_grad(self, set_to_none: bool = True):
        super().zero_grad(set_to_none)
        for live in self._live:
            if set_to_none:
                live.grad = None
            elif live.grad is not None:
                live.grad.zero_()


def make_optimizer(model: torch.nn.Module, cfg: CommonTrainConfig) -> torch.optim.Optimizer:
    lora, new = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (lora if "lora_" in name else new).append(p)
    groups = [{"params": lora, "lr": cfg.lr},
              {"params": new, "lr": cfg.lr_new_modules}]
    cls = Fp32MasterAdamW if cfg.fp32_master else torch.optim.AdamW
    return cls(groups, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))


def make_scheduler(opt: torch.optim.Optimizer, cfg: CommonTrainConfig):
    def fn(step: int) -> float:
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
        p = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * min(p, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


_SAC_INFIX = "._checkpoint_wrapped_module"


def canon_param_name(name: str) -> str:
    """state_dict()-style name for a named_parameters() entry.

    cosmos wraps every DiT block in an activation-checkpoint wrapper on
    TRAINING builds; `named_parameters()` reports those params as
    `net.blocks.N._checkpoint_wrapped_module.self_attn...` while
    `state_dict()` (what checkpoints store and what load_state_dict wants)
    strips the infix. Keying the EMA on the raw names meant `--ema` matched
    0/560 LoRA tensors on load and silently deployed raw LoRA + EMA heads
    (found 2026-08-26; v4 impact ~1e-3 rel-L2 because its LR had decayed to
    0, but a fine-tune with a live LR would have been evaluated wrong)."""
    return name.replace(_SAC_INFIX, "")


class EMA:
    """EMA over trainable params only, keyed by canonical (state_dict) names."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {canon_param_name(n): p.detach().clone().float()
                       for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for n, p in model.named_parameters():
            n = canon_param_name(n)
            if n in self.shadow:
                self.shadow[n].lerp_(p.detach().float(), 1.0 - self.decay)

    def state_dict(self) -> dict:
        return self.shadow

    def load_state_dict(self, sd: dict) -> None:
        sd = {canon_param_name(k): v for k, v in sd.items()}   # pre-fix ckpts
        missing = [n for n in self.shadow if n not in sd]
        assert not missing, f"EMA resume: shadow keys absent from checkpoint: {missing[:8]}"
        for n in self.shadow:
            self.shadow[n].copy_(sd[n].float())


# ---------------------------------------------------------------------------
# checkpoints — never re-save the frozen 2B base
# ---------------------------------------------------------------------------

def trainable_state_dicts(model: torch.nn.Module) -> tuple[dict, dict]:
    """(lora, phantom) — the only weights PHANTOM checkpoints carry."""
    lora, phantom = {}, {}
    for name, t in model.state_dict().items():
        if "lora_" in name:
            lora[name] = t
        elif "phantom_" in name:
            phantom[name] = t
    return lora, phantom


def save_phantom_checkpoint(path: Path, model: torch.nn.Module, *,
                            hw: HardwareConfig, bb: BackboneConfig,
                            mc: PhantomModelConfig, train_cfg: CommonTrainConfig,
                            step: int, base_ckpt_path: str = "",
                            norm_stats: NormStats | None = None,
                            optimizer=None, scheduler=None, ema: EMA | None = None,
                            text_conditioning: dict | None = None) -> None:
    lora, phantom = trainable_state_dicts(model)
    payload = {
        "format_version": CKPT_FORMAT_VERSION,
        "base_ckpt_path": base_ckpt_path,
        "lora": lora,
        "phantom_modules": phantom,
        "ema": ema.state_dict() if ema else None,
        "configs": {
            "hardware_shapes": hw.shape_relevant_fields(),
            "hardware_hash": hw.config_hash(),
            "backbone": bb.to_dict(),
            "model": mc.to_dict(),
            "train": train_cfg.to_dict(),
            "text_conditioning": text_conditioning or {},
        },
        "norm_stats": ({"mean": {k: v.tolist() for k, v in norm_stats.mean.items()},
                        "std": {k: v.tolist() for k, v in norm_stats.std.items()}}
                       if norm_stats else None),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "step": step,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    log.info("checkpoint saved: %s (step %d, lora %d keys, phantom %d keys)",
             path, step, len(lora), len(phantom))


def assert_model_config_matches(payload: dict, model: torch.nn.Module, *,
                                tolerate: frozenset[str] | Iterable[str] = frozenset()) -> None:
    """Refuse a checkpoint whose saved PhantomModelConfig differs from the one
    the model was BUILT with (P10B, review 2026-08-28).

    Behavioral knobs live in `mc`, not in the weights: `rope_time_mode`
    decides the ACTION frames' RoPE positions (recomputed at runtime from mc
    in phantom_dit.py — no saved tensor can rescue a mismatch) and
    `acc.self_anticipation` decides whether the gate sees its own prediction
    or leaked GT. Loading a `time_true`/`two_pass` checkpoint into a
    default-built (`aligned`/`gt_noised`) model raised nothing and silently
    shifted every action frame's temporal phase — distill_hid, dagger.relabel
    and finetune_hids all did exactly that.

    `student` is the ONE legitimate difference: teacher->student init loads a
    teacher checkpoint into a student model on purpose (allow_missing=True).
    `mask_wrist` is likewise an INPUT-ABLATION switch applied on top of a
    trained checkpoint (vision_only / drop_tactile), not a weight-layout knob.

    `TRAIN_ONLY_MODEL_FIELDS` (the FT-A contact-loss knobs) are ignored too:
    they shape the TRAINING objective only — nothing in the deployed forward
    or the sampler reads them, so a checkpoint fine-tuned with a different
    contact-loss balance is the same model at deploy. `action_noise_per_strip`
    is deliberately NOT in that set: it changes `sample()`.

    `tolerate` widens that ignore set for ONE caller class: a fine-tune that
    deliberately flips a training-objective knob relative to the checkpoint it
    initializes from (`train_teacher --init-weights` passes
    `FINETUNE_MUTABLE_MODEL_FIELDS`; validation 2026-08-30 F1). Every other
    caller — `--resume`, deploy, distill, replay — stays strict, and the
    fine-tune path re-checks the same fields itself via `model_config_drift`
    so the difference is warned about rather than silently accepted.
    """
    saved = (payload.get("configs") or {}).get("model")
    mc = getattr(model, "mc", None)          # PhantomRectifiedFlow.mc
    if not isinstance(saved, dict) or mc is None:
        return
    ignore = ("student", "mask_wrist", *TRAIN_ONLY_MODEL_FIELDS, *tolerate)
    cur = {k: v for k, v in mc.to_dict().items() if k not in ignore}
    drift = {k: {"checkpoint": saved.get(k), "model": v}
             for k, v in cur.items() if k in saved and saved[k] != v}
    if drift:
        raise RuntimeError(
            f"model-config drift vs checkpoint: {drift} — the model was built "
            f"with different behavior than these weights were trained with. "
            f"Build it from the checkpoint's own config: "
            f"mc = PhantomModelConfig.from_dict(payload['configs']['model']) "
            f"(dataclasses.replace(mc, student=True) for the student).")


def load_phantom_checkpoint(path: Path, model: torch.nn.Module, *,
                            hw: HardwareConfig, load_ema: bool = False,
                            allow_missing: bool = False,
                            payload: dict | None = None,
                            tolerate_model_fields: frozenset[str] | Iterable[str] = frozenset()) -> dict:
    """Load lora+phantom weights into a built model; asserts hardware
    shape-compat (value-only drift warns via hash). `payload` lets callers
    that already torch.load'ed the file (e.g. to reconstruct the saved model
    config BEFORE building — see run_deploy) skip the second 286MB read.
    `tolerate_model_fields` is forwarded to `assert_model_config_matches` —
    only the `--init-weights` fine-tune path passes it."""
    if payload is None:
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
    assert payload["format_version"] == CKPT_FORMAT_VERSION
    saved_shapes = payload["configs"]["hardware_shapes"]
    cur_shapes = hw.shape_relevant_fields()
    mismatches = {k: (saved_shapes.get(k), v) for k, v in cur_shapes.items()
                  if _norm_shape(saved_shapes.get(k)) != _norm_shape(v)}
    assert not mismatches, (
        f"hardware shape-relevant fields changed since this checkpoint was trained: "
        f"{mismatches} — retrain or restore configs/hardware.yaml")
    if payload["configs"]["hardware_hash"] != hw.config_hash():
        log.warning("hardware config VALUES differ from checkpoint provenance "
                    "(shape-compatible — proceeding)")
    assert_model_config_matches(payload, model, tolerate=tolerate_model_fields)
    weights = dict(payload["lora"])
    weights.update(payload["phantom_modules"])
    if load_ema and payload.get("ema"):
        ema_sd = {canon_param_name(k): v for k, v in payload["ema"].items()}
        hit = {k: v for k, v in ema_sd.items() if k in weights}
        n_lora = sum("lora_" in k for k in hit)
        assert len(hit) == len(ema_sd) and (n_lora > 0 or not any("lora_" in k for k in weights)), (
            f"EMA tensors do not map onto the model: matched {len(hit)}/{len(ema_sd)} "
            f"({n_lora} LoRA) — checkpoint/model naming drift; refuse rather than "
            f"silently deploying raw weights")
        weights.update(hit)
        log.info("EMA weights applied: %d tensors (%d LoRA)", len(hit), n_lora)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    unexpected = [k for k in unexpected]
    if unexpected and not allow_missing:
        raise RuntimeError(f"checkpoint keys unknown to this model: {unexpected[:8]}")
    not_loaded = [k for k in missing if "lora_" in k or "phantom_" in k]
    if not_loaded and not allow_missing:
        raise RuntimeError(f"model trainable keys absent from checkpoint: {not_loaded[:8]} "
                           "(pass allow_missing=True for teacher->student init)")
    if allow_missing and (unexpected or not_loaded):
        log.info("partial init (teacher->student): %d checkpoint keys dropped "
                 "(teacher-only paths), %d model keys left fresh",
                 len(unexpected), len(not_loaded))
    return payload


def _norm_shape(v):
    if isinstance(v, (list, tuple)):
        return tuple(_norm_shape(x) for x in v)
    return v


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------

CLOSE_ABS_POS = 0.45      # gripper position that unambiguously means "closed"
CLOSE_ABS_RISE = 0.15     # ... after rising this much from the running minimum
CLOSE_MAX_DELTA = 0.05    # fallback: within this of the episode's tightest closure
CLOSE_MAX_RISE = 0.10


def close_index(pos: np.ndarray) -> int | None:
    """Index of the first gripper close, or None if the gripper never closes.

    Primary rule: pos > 0.45 after rising > 0.15 from its running minimum.
    Fallback (only when the primary never fires): first sample within 0.05
    of the episode's tightest closure after rising > 0.10 — wide grasps
    (Carton closes to 0.40-0.47) never cross 0.45 and were silently left
    unweighted by --grasp-frac (24% of Carton demos). Lowering the global
    threshold instead fires on pre-grasp aperture adjustments (egg: 108/250
    episodes > 1s early), so the fallback is used strictly as a fallback.
    Validated on 1000 success demos: identical to the primary wherever it
    fires, Carton coverage 76% -> 99%, egg/waffles/whiteboard 100%."""
    pos = np.asarray(pos, dtype=np.float64)
    if pos.size == 0:
        return None
    run_min = np.minimum.accumulate(pos)
    hit = np.nonzero((pos > CLOSE_ABS_POS) & (pos - run_min > CLOSE_ABS_RISE))[0]
    if len(hit):
        return int(hit[0])
    hit = np.nonzero((pos >= pos.max() - CLOSE_MAX_DELTA)
                     & (pos - run_min > CLOSE_MAX_RISE))[0]
    return int(hit[0]) if len(hit) else None


def manifest_split(data_root: Path, split: str) -> list[Path] | None:
    """Episode dirs for `split` from manifests/all.jsonl, or None if absent.

    The dataset ships a manifest next to the task folders (data_root is
    <root>/tasks, so the manifest is ../manifests/all.jsonl). Without this the
    training set silently includes the held-out episodes and no validation
    number means anything."""
    if split == "all":
        return None
    mf = Path(data_root).parent / "manifests" / "all.jsonl"
    if not mf.exists():
        log.warning("no manifest at %s — using every episode under %s "
                    "(NO held-out set)", mf, data_root)
        return None
    rows = [json.loads(l) for l in mf.read_text().splitlines() if l.strip()]
    # the manifest is the ONLY thing standing between the data and the
    # optimizer on this path (list_episodes() is bypassed), so it must be a
    # bijection onto finalized episodes: no duplicate rows, no path in two
    # splits, no missing directories (Codex review 2026-08-26)
    by_path: dict[str, str] = {}
    for r in rows:
        prev = by_path.get(r["path"])
        if prev is not None and prev != r.get("split"):
            raise SystemExit(f"manifest {mf}: {r['path']} appears in both "
                             f"{prev!r} and {r.get('split')!r} splits")
        by_path[r["path"]] = r.get("split")
    if len(by_path) != len(rows):
        dups = sorted({r["path"] for r in rows if sum(x["path"] == r["path"] for x in rows) > 1})
        raise SystemExit(f"manifest {mf}: duplicate rows for {dups[:5]}")
    eps: list[Path] = []
    missing: list[str] = []
    excluded: list[str] = []
    for r in rows:
        if r.get("split") != split:
            continue
        p = Path(data_root).parent / r["path"]
        if not (p / "meta.json").exists():
            missing.append(r["path"])
            continue
        meta_d = json.loads((p / "meta.json").read_text())
        # P9 (review 2026-08-28): tags disqualify BEFORE the status check —
        # 'unlabeled' episodes are deliberately status='aborted' (that is how
        # the no-verdict guard files them), so the hard error below would turn
        # a correctly-excluded episode into a crashed run.
        tags = {str(t) for t in (meta_d.get("tags") or [])}
        tags |= {str(t) for t in (r.get("tags") or [])}
        bad = sorted(tags & set(NON_TRAINING_TAGS))
        if bad:
            excluded.append(f"{r['path']} ({','.join(bad)})")
            continue
        st = meta_d.get("status", "finalized")
        if st != "finalized":
            raise SystemExit(f"manifest {mf}: {r['path']} has status {st!r} — "
                             f"non-finalized episodes must not train")
        # meta.json written before the status field existed has no key; the
        # status check above already read it as finalized, so keep that
        # reading (EpisodeMeta defaults to "recording").
        meta = EpisodeMeta.from_dict({**meta_d, "status": st})
        if not is_trainable_episode(meta):
            # what is left after the two checks above: an UNJUDGED POLICY
            # ROLLOUT (policy != teleop, success is None). Loud, not skipped —
            # it means a DAgger intake pass admitted rollouts nobody labelled,
            # and every one of them would ground the student's own mistakes at
            # action_weight 1.0.
            raise SystemExit(
                f"manifest {mf}: {r['path']} is a policy rollout "
                f"(policy={meta.policy!r}) with no success verdict — label it "
                f"(run_deploy's prompt) or drop it from the manifest; "
                f"unjudged rollouts must never train")
        eps.append(p)
    if excluded:
        log.warning("manifest split %r: %d episodes excluded by tag "
                    "(%s%s)", split, len(excluded), ", ".join(excluded[:3]),
                    ", ..." if len(excluded) > 3 else "")
    if missing:
        raise SystemExit(f"manifest {mf}: {len(missing)} {split!r} rows point at missing "
                         f"episodes, e.g. {missing[:3]} — partial dataset")
    log.info("manifest split %r: %d episodes", split, len(eps))
    return eps


class WindowDataset(Dataset):
    """Wraps WindowSampler over an episode root for DataLoader consumption."""

    def __init__(self, root: Path, sampler, windows_per_episode: int = 8,
                 episodes: list[Path] | None = None, resample: bool = True,
                 seed: int = 0, grasp_frac: float = 0.0,
                 grasp_window_s: tuple[float, float] = (1.5, 0.2),
                 photo_aug: float = 0.0, commit_band_weight: float = 1.0):
        self.sampler = sampler
        self.index = sampler.build_index(root, windows_per_episode, episodes)
        # A frozen index replays the same anchors every epoch (~35x over a long
        # run) — redraw t0 within the episode's admissible range on each access
        # so the run keeps seeing fresh windows from the same episodes. Held-out
        # sets pass resample=False to stay comparable across evals.
        self.resample = resample
        self.seed = int(seed)
        self.epoch = 0                     # set by train_loop at each rollover
        self._rng = np.random.default_rng(seed)
        # Terminal-phase weighting (rig postmortem 2026-08-20: the policy
        # under-executes the last ~6 cm — closes 35-80 mm above grasp height
        # in 13/13 rig episodes). With probability grasp_frac a window's t0 is
        # drawn from [t_close - grasp_window_s[0], t_close - grasp_window_s[1]]
        # so the chunk spans the commit phase. 0 = uniform (unchanged).
        self.grasp_frac = float(grasp_frac)
        self.grasp_window_s = grasp_window_s
        self._close_cache: dict[Path, float | None] = {}
        # Photometric augmentation of the SCENE camera only (train sets):
        # rig session 08-18/20 ran under noticeably different lighting than
        # collection; the camera is the model's dominant input (conditioning
        # ablation). One jitter per window (lighting is constant within an
        # episode), same jitter for every frame of the window.
        self.photo_aug = float(photo_aug)
        # D8's "x2 window multiplier in the commit band" (validation
        # 2026-08-30 F19): an extra ACTION-loss multiplier for windows whose
        # t0 lands inside the same pre-close band `grasp_frac` anchors in.
        # 1.0 = off (the default); it composes with the per-episode
        # EpisodeMeta.weight and never rescues a failure demo's 0.
        self.commit_band_weight = float(commit_band_weight)

    def __len__(self) -> int:
        return len(self.index)

    def _close_time(self, ep: Path) -> float | None:
        """First gripper-close time (stream ts base) or None if never closes."""
        if ep not in self._close_cache:
            t = None
            try:
                import zarr
                g = zarr.open(str(Path(ep) / "gripper.zarr"), mode="r")
                pos = np.asarray(g["data"][:, 0], dtype=np.float64)
                ts = np.asarray(g["ts"][:], dtype=np.float64)
                i = close_index(pos)
                if i is not None:
                    t = float(ts[i])
            except Exception as e:          # noqa: BLE001
                log.warning("grasp weighting: no close time for %s (%s)", ep, e)
            self._close_cache[ep] = t
        return self._close_cache[ep]

    def grasp_coverage(self) -> dict:
        """How many indexed episodes can actually be grasp-weighted (close
        time found AND the pre-close band intersects the valid range). Log
        this: a fine-tune whose weighting silently degraded to uniform would
        otherwise look identical in the training log."""
        eps = {wi.episode: (wi.lo, wi.hi) for wi in self.index}
        ok = no_close = no_band = 0
        for ep, (lo, hi) in eps.items():
            tc = self._close_time(ep)
            if tc is None:
                no_close += 1
            elif min(hi, tc - self.grasp_window_s[1]) <= max(lo, tc - self.grasp_window_s[0]):
                no_band += 1
            else:
                ok += 1
        return {"episodes": len(eps), "weightable": ok, "no_close": no_close,
                "band_outside_range": no_band}

    def __getitem__(self, i: int) -> dict:
        wi = self.index[i]
        t0 = wi.t0
        if self.resample and wi.hi > wi.lo:
            t0 = float(self._rng.uniform(wi.lo, wi.hi))
            if self.grasp_frac > 0 and self._rng.uniform() < self.grasp_frac:
                tc = self._close_time(wi.episode)
                if tc is not None:
                    a = max(wi.lo, tc - self.grasp_window_s[0])
                    b = min(wi.hi, tc - self.grasp_window_s[1])
                    if b > a:
                        t0 = float(self._rng.uniform(a, b))
        item = self.sampler.sample(wi.episode, t0)
        if self.photo_aug > 0 and self.resample and "video" in item:
            s = self.photo_aug
            v = item["video"]                       # (T, 3, H, W) in [-1, 1]
            x = (v + 1.0) * 0.5                     # -> [0, 1]
            gain = 1.0 + s * float(self._rng.uniform(-0.3, 0.3))
            contrast = 1.0 + s * float(self._rng.uniform(-0.25, 0.25))
            ch = 1.0 + s * self._rng.uniform(-0.08, 0.08, size=3)
            import torch as _t
            x = ((x - 0.5) * contrast + 0.5) * gain \
                * _t.as_tensor(ch, dtype=x.dtype).view(1, 3, 1, 1)
            item["video"] = (x.clamp(0.0, 1.0) * 2.0 - 1.0).to(v.dtype)
        if self.commit_band_weight != 1.0 and item.get("action_weight", 0.0):
            tc = self._close_time(wi.episode)
            if tc is not None and \
                    tc - self.grasp_window_s[0] <= t0 <= tc - self.grasp_window_s[1]:
                item["action_weight"] = float(item["action_weight"]) * self.commit_band_weight
        return item


def collate_windows(items: list[dict]) -> dict:
    out: dict = {}
    for k in items[0]:
        v = items[0][k]
        if torch.is_tensor(v):
            out[k] = torch.stack([it[k] for it in items])
        else:
            out[k] = [it[k] for it in items]
    return out


def ensure_synthetic_dataset(root: Path, hw: HardwareConfig, n_episodes: int = 4,
                             duration_s: float = 8.0, seed: int = 0) -> Path:
    """Generate (once per hardware-config hash) a synthetic episode set for
    --synthetic smoke runs. Episodes live under root/<hash>/ so that changing
    hardware.yaml never silently reuses data generated under old values.
    Under torchrun only rank 0 generates (the barrier holds the other ranks
    until the episodes exist); identical behavior single-process."""
    from phantom.data.synthetic import SyntheticEpisodeGenerator
    root = Path(root) / hw.config_hash()[:10]
    if is_main() and not list(root.rglob("meta.json")):
        root.mkdir(parents=True, exist_ok=True)
        SyntheticEpisodeGenerator(hw, seed=seed).generate_dataset(
            root, n_episodes=n_episodes, duration_s=duration_s)
    if dist.is_initialized():
        dist.barrier()
    return root


def make_loader(ds: Dataset, cfg: CommonTrainConfig, *,
                collate_fn=collate_windows, shuffle: bool | None = None) -> DataLoader:
    """The one training DataLoader for all four programs. world==1 reproduces
    the historical construction exactly (shuffle=True, drop_last=True,
    synthetic -> workers 0); under torchrun a DistributedSampler shards the
    window index disjointly per rank (train_loop's re-iteration calls
    set_epoch for the reshuffle). Construct AFTER any ds.index mutation
    (distill_hid --extra-data) — the sampler snapshots len(ds)."""
    rank, world = setup_ddp()          # idempotent
    sampler = None
    if world > 1:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(ds, num_replicas=world, rank=rank,
                                     shuffle=True, seed=cfg.seed, drop_last=True)
        assert len(sampler) >= cfg.batch_size, (
            f"dataset too small to shard: {len(ds)} windows over {world} ranks "
            f"gives {len(sampler)}/rank < batch_size {cfg.batch_size}")
    return DataLoader(ds, batch_size=cfg.batch_size,
                      shuffle=((sampler is None) if shuffle is None else shuffle),
                      sampler=sampler,
                      num_workers=0 if cfg.synthetic else cfg.num_workers,
                      collate_fn=collate_fn, drop_last=True,
                      worker_init_fn=_seed_worker_rng)


def _seed_worker_rng(worker_id: int) -> None:
    """Give every DataLoader worker its own numpy stream per epoch.

    Workers are forked with a COPY of the dataset's `_rng`, so without this
    all workers drew the same t0 / grasp-anchor / photo-aug jitters in
    lockstep and replayed them every epoch (measured 32% distinct jitters at
    16 workers, 2026-08-26). torch's own per-worker seeding does not cover
    numpy Generators held on the dataset."""
    info = torch.utils.data.get_worker_info()
    ds = info.dataset if info is not None else None
    if ds is not None and hasattr(ds, "_rng"):
        ds._rng = np.random.default_rng(
            [int(getattr(ds, "seed", 0)), int(getattr(ds, "epoch", 0)), int(worker_id)])


# ---------------------------------------------------------------------------
# generic loop
# ---------------------------------------------------------------------------

def evaluate(model: torch.nn.Module, val_loader: DataLoader, step_fn,
             max_batches: int = 24) -> dict[str, float]:
    """Mean losses over a held-out loader (no grad). Keeps the run honest:
    without this the only signal for days is the training loss."""
    was_training = model.training
    model.eval()
    sums: dict[str, float] = {}
    n = 0
    # stride so max_batches SPAN the whole ordered val set: the manifest is
    # task-sorted, so taking the first N batches would silently measure only
    # the first task's episodes and never see the rest
    stride = max(1, len(val_loader) // max_batches)
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i % stride != 0:
                continue
            if n >= max_batches:
                break
            parts = step_fn(batch)
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + float(v)
            n += 1
    if was_training:
        model.train()
    return {k: v / max(n, 1) for k, v in sums.items()}


def label_tally(ds, n_windows: int = 48) -> tuple[dict, dict]:
    """Sample real windows and tally (gate_counts, event_shares)."""
    from collections import Counter
    from phantom.config.model import EVENTS
    stride = max(1, len(ds) // n_windows)
    gates: Counter = Counter()
    events: Counter = Counter()
    for i in list(range(0, len(ds), stride))[:n_windows]:
        w = ds[i]
        gates[int(float(w["gate_label"]) > 0.5)] += 1
        for e in w["events"].tolist():
            events[EVENTS[e]] += 1
    tot = sum(events.values()) or 1
    return dict(gates), {k: v / tot for k, v in events.items()}


def assert_label_sanity(ds, logger, n_windows: int = 48,
                        min_event_classes: int = 3,
                        min_share: float = 0.005) -> None:
    """Hard-fail on degenerate contact labels BEFORE burning GPU time.

    Requires both gate classes present and >= min_event_classes event
    classes above min_share. Teacher v3 trained on 100%-positive gates and
    96% 'hold' events (single-pixel contact saturation, issue #1) and
    nothing was loud about it."""
    gates, ev = label_tally(ds, n_windows)
    logger.info("label sanity: gate %s | events %s", dict(gates),
                {k: f"{v:.1%}" for k, v in sorted(ev.items())})
    problems = []
    if len(gates) < 2:
        problems.append(f"gate labels are constant ({dict(gates)})")
    rich = [k for k, v in ev.items() if v >= min_share]
    if len(rich) < min_event_classes:
        problems.append(
            f"only {len(rich)} event classes above {min_share:.1%} ({rich})")
    if problems:
        raise SystemExit(
            "DEGENERATE LABELS: " + "; ".join(problems) + " — the contact/"
            "anticipation stack would train on constants (issue #1). Check "
            "derived.tau_contact_depth/tau_contact_area against this "
            "dataset before training.")


def evaluate_sampled(rf, val_ds, norm_action_mean, norm_action_std,
                     n_windows: int = 8, nfe: int = 5,
                     seed: int = 123) -> dict[str, float]:
    """Open-loop SAMPLED-chunk metrics on held-out windows — what the flow
    loss cannot see. teacher v3 shipped with val_action_v_mse pinned at its
    no-conditioning floor while sampled actions were behaviorally wrong; this
    is the standing guard against that (v4 audit 2026-08-14).

    Returns mag_ratio (sampled/GT |dpos| — 1.0 is demo vigor), dir_cosine
    (chunk direction vs GT), sampled_mse (normalized action space)."""
    was_training = rf.training
    rf.eval()
    stride = max(1, len(val_ds) // n_windows)
    idxs = list(range(0, len(val_ds), stride))[:n_windows]
    mean = np.asarray(norm_action_mean, dtype=np.float64)
    std = np.asarray(norm_action_std, dtype=np.float64)
    mags_p, mags_g, coss, mses = [], [], [], []
    gen_state = rf._gen.get_state()
    with torch.no_grad():
        for j, i in enumerate(idxs):
            item = val_ds[i]
            batch = {}
            for k, v in item.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.unsqueeze(0)
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.from_numpy(v).unsqueeze(0)
                elif isinstance(v, str):
                    batch[k] = [v]
                elif isinstance(v, (int, float, np.floating)):
                    batch[k] = torch.tensor([v])
                else:
                    batch[k] = v
            for k, v in batch.items():
                if torch.is_tensor(v):
                    v = v.to(rf.device)
                    if v.is_floating_point():
                        v = v.to(rf.dtype)
                    batch[k] = v
            # no privileged future: ACC's prev-cpk summary/events come from
            # the batch's GT contact package at build_x0 time — deploy never
            # has that, and the whole point of this metric is deploy realism
            for k in list(batch):
                if k == "events" or k.startswith("cpk_"):
                    batch[k] = torch.zeros_like(batch[k])
            rf._gen = torch.Generator().manual_seed(seed + j)
            pred = rf.sample(batch, nfe=nfe)
            gt = batch["action_chunk"][0].float().cpu().numpy().astype(np.float64)
            pr = pred.actions_B_H_A[0].float().cpu().numpy().astype(np.float64)
            mses.append(float(((pr - gt) ** 2).mean()))
            gt_d, pr_d = gt * std + mean, pr * std + mean
            mags_g.append(float(np.linalg.norm(gt_d[:, :3], axis=1).mean()))
            mags_p.append(float(np.linalg.norm(pr_d[:, :3], axis=1).mean()))
            den = (np.linalg.norm(pr_d[:, :3]) * np.linalg.norm(gt_d[:, :3]) + 1e-12)
            coss.append(float(np.dot(pr_d[:, :3].ravel(), gt_d[:, :3].ravel()) / den))
    rf._gen = torch.Generator()
    rf._gen.set_state(gen_state)
    if was_training:
        rf.train()
    return {
        "sampled_mag_ratio": float(np.mean(mags_p) / (np.mean(mags_g) + 1e-9)),
        "sampled_dir_cosine": float(np.mean(coss)),
        "sampled_action_mse": float(np.mean(mses)),
    }


class _StepModule(torch.nn.Module):
    """Makes a program's step_fn the FORWARD of a module that owns the
    program's parameters, so DistributedDataParallel can wrap it: DDP arms
    its gradient all-reduce in forward(), and step_fn keeps calling the
    inner module's training_step exactly as before."""

    def __init__(self, inner: torch.nn.Module, step_fn):
        super().__init__()
        self.inner = inner
        self._step_fn = step_fn

    def forward(self, batch):
        return self._step_fn(batch)


def train_loop(cfg: CommonTrainConfig, model: torch.nn.Module, loader: DataLoader,
               step_fn, *, on_checkpoint=None, val_loader: DataLoader | None = None,
               eval_step_fn=None, sampled_eval_fn=None,
               resume_payload: dict | None = None) -> int:
    """step_fn(batch) -> dict with 'total' loss tensor. Handles grad accum,
    clipping, EMA, logging; returns final step.

    val_loader: optional held-out loader evaluated every cfg.eval_every steps
    (eval_step_fn defaults to step_fn).

    resume_payload: a save_phantom_checkpoint payload — restores optimizer,
    scheduler, EMA and the step counter (weights are the caller's job, via
    load_phantom_checkpoint, BEFORE calling this). Fp32MasterAdamW masters
    re-init from the restored bf16 weights: the sub-ulp fp32 residual is lost
    once at the resume point (one extra bf16 rounding), the exp_avg/exp_avg_sq
    moments load exactly."""
    rank, world = setup_ddp()
    device = cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
    model = model.to(device)
    base = model                       # the program's module (checkpoints, EMA)
    run_step, sync_ctx = step_fn, None
    if world > 1:
        # DDP only all-reduces gradients for a backward whose FORWARD went
        # through the wrapper. Every program's step_fn calls its own module's
        # training_step directly, so the wrapper below routes that call
        # through DDP.forward; calling step_fn on the bare module (the old
        # code) gave N independent single-GPU runs (found 2026-08-26).
        ddp = torch.nn.parallel.DistributedDataParallel(
            _StepModule(base, step_fn), find_unused_parameters=True)
        run_step, sync_ctx = ddp, ddp.no_sync
        model = ddp
    opt = make_optimizer(model, cfg)
    sched = make_scheduler(opt, cfg)
    ema = EMA(base, cfg.ema_decay)

    start_step = 0
    if resume_payload is not None:
        assert resume_payload.get("optimizer"), "resume checkpoint has no optimizer state"
        opt.load_state_dict(resume_payload["optimizer"])
        if resume_payload.get("scheduler"):
            sched.load_state_dict(resume_payload["scheduler"])
        if resume_payload.get("ema"):
            ema.load_state_dict(resume_payload["ema"])
        start_step = int(resume_payload["step"])
        if is_main():
            log.info("resumed optimizer/scheduler/ema at step %d (lr %.3g)",
                     start_step, sched.get_last_lr()[0])

    # offset by start_step so a resumed run does not replay the step-0 data
    # order / noise stream it already trained on
    torch.manual_seed(cfg.seed + rank + start_step)
    np.random.seed(cfg.seed + rank + start_step)
    # the rectified-flow model keeps its own Generator for (t, eps, cond-drop)
    # draws; it was seeded to 0 regardless of cfg.seed/rank, so DDP ranks
    # shared one noise stream (found 2026-08-26)
    if hasattr(base, "_gen"):
        base._gen.manual_seed(cfg.seed + 1000 * rank + start_step)

    step, t0 = start_step, time.perf_counter()
    epoch = 0
    it = iter(loader)
    while step < cfg.max_steps:
        opt.zero_grad(set_to_none=True)
        logs: dict[str, float] = {}
        for micro in range(cfg.grad_accum):
            try:
                batch = next(it)
            except StopIteration:
                epoch += 1
                if hasattr(getattr(loader, "sampler", None), "set_epoch"):
                    loader.sampler.set_epoch(epoch)   # DistributedSampler reshuffle
                if hasattr(loader.dataset, "epoch"):
                    loader.dataset.epoch = epoch      # fresh per-worker numpy streams
                it = iter(loader)
                batch = next(it)
            # under DDP skip the all-reduce on every micro-batch but the last
            last = micro == cfg.grad_accum - 1
            with (sync_ctx() if (sync_ctx is not None and not last) else contextlib.nullcontext()):
                parts = run_step(batch)
                (parts["total"] / cfg.grad_accum).backward()
            for k, v in parts.items():
                if torch.is_tensor(v) and v.ndim == 0:
                    logs[k] = logs.get(k, 0.0) + float(v.detach()) / cfg.grad_accum
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], cfg.grad_clip)
        opt.step()
        sched.step()
        ema.update(base)
        step += 1
        if step % cfg.log_every == 0:
            if world > 1 and logs:
                # true cross-rank mean; SUM/world (ReduceOp.AVG is NCCL-only)
                keys = sorted(logs)
                t = torch.tensor([logs[k] for k in keys],
                                 device=device if device != "cpu" else None)
                dist.all_reduce(t)
                logs = {k: float(v) / world for k, v in zip(keys, t.tolist())}
            if is_main():
                rate = (step - start_step) / (time.perf_counter() - t0)
                log.info("step %d/%d  %s  (%.2f it/s)", step, cfg.max_steps,
                         "  ".join(f"{k}={v:.4f}" for k, v in sorted(logs.items())), rate)
        if (val_loader is not None and is_main() and cfg.eval_every
                and step % cfg.eval_every == 0):
            log.info("EVAL running at step %d (val%s)...", step,
                     " + sampled" if sampled_eval_fn is not None else "")
            # eval calls stochastic training_step / sample — snapshot the
            # model's generator so evaluation never perturbs the training
            # noise stream (runs stay comparable across eval_every settings)
            _gen = getattr(model, "_gen", None)
            _gen_state = _gen.get_state() if _gen is not None else None
            vm = evaluate(model, val_loader, eval_step_fn or step_fn)
            log.info("EVAL step %d  %s", step,
                     "  ".join(f"val_{k}={v:.4f}" for k, v in sorted(vm.items())))
            if sampled_eval_fn is not None:
                sm = sampled_eval_fn()
                log.info("EVAL-SAMPLED step %d  %s", step,
                         "  ".join(f"{k}={v:.4f}" for k, v in sorted(sm.items())))
            if _gen_state is not None:
                model._gen.set_state(_gen_state)
        if is_main() and on_checkpoint and step % cfg.ckpt_every == 0:
            on_checkpoint(step, opt, sched, ema)
    if is_main() and on_checkpoint:
        on_checkpoint(step, opt, sched, ema)
    if world > 1:
        dist.destroy_process_group()
    return step


def to_device(batch: dict, device, dtype=None) -> dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            v = v.to(device)
            if dtype is not None and v.is_floating_point():
                v = v.to(dtype)
        out[k] = v
    return out
