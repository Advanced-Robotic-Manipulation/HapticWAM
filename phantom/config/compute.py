"""Compute-target configuration for TRAINING.

configs/compute.yaml names the active profile (`target:`) — THE file to edit
when moving training between the single-5090 box and an 8xH100/8xA100 node.
configs/compute.local.yaml (gitignored) is merged over it per machine, so a
cluster checkout never edits the committed default. Deploy / eval / data
collection are unaffected by this module.

All override fields are Optional: None = keep the program's own dataclass
default. The `rtx5090` profile overrides nothing at all, which is what makes
the historical single-GPU behavior provably identical (the four programs have
DIFFERENT defaults — tactile batch 32 vs teacher/HID batch 1 x accum 8 — so a
default profile with hard values would clobber one of them).

Precedence per value: dataclass defaults < profile (programs: sub-overrides
beat profile level) < explicit CLI flags.

Torch-free on purpose (the config package never imports torch); the
dtype-string -> torch.dtype mapping lives in phantom.train.common.pick_dtype.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

log = logging.getLogger(__name__)

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
DEFAULT_COMPUTE_YAML = CONFIGS_DIR / "compute.yaml"
LOCAL_COMPUTE_YAML = CONFIGS_DIR / "compute.local.yaml"

# 1:1 CommonTrainConfig field names (dtype is separate — not a dataclass field)
_CFG_FIELDS = ("batch_size", "grad_accum", "num_workers",
               "lr", "lr_new_modules", "warmup_steps")
_OVERRIDE_FIELDS = ("dtype",) + _CFG_FIELDS

ProgramName = Literal["pretrain_tactile", "train_teacher",
                      "distill_hid", "finetune_hids"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ProgramOverrides(_Frozen):
    """None = don't touch the program's dataclass default."""
    dtype: Literal["bf16", "fp32"] | None = None   # consumed by C.pick_dtype
    batch_size: int | None = Field(default=None, gt=0)   # per GPU
    grad_accum: int | None = Field(default=None, gt=0)
    num_workers: int | None = Field(default=None, ge=0)
    lr: float | None = Field(default=None, gt=0)
    lr_new_modules: float | None = Field(default=None, gt=0)
    warmup_steps: int | None = Field(default=None, ge=0)

    def apply_to(self, cfg):
        """Profile values over a frozen train-config dataclass. CLI flags are
        applied AFTER this by the programs' apply_overrides, so they win."""
        updates = {f: getattr(self, f) for f in _CFG_FIELDS
                   if getattr(self, f) is not None}
        return dataclasses.replace(cfg, **updates) if updates else cfg


class ComputeProfile(ProgramOverrides):
    description: str = ""
    launcher: Literal["python", "torchrun"] = "python"   # declarative: what YOU run
    nproc_per_node: int = Field(default=1, ge=1)
    programs: dict[ProgramName, ProgramOverrides] = {}

    def for_program(self, program: str) -> ProgramOverrides:
        merged = {f: getattr(self, f) for f in _OVERRIDE_FIELDS}
        sub = self.programs.get(program)
        if sub is not None:
            merged.update({f: getattr(sub, f) for f in _OVERRIDE_FIELDS
                           if getattr(sub, f) is not None})
        return ProgramOverrides(**merged)

    def check_world(self, world: int) -> None:
        """Warn-only: profile/launch mismatch (e.g. h100x8 without torchrun).
        Per-GPU batch/accum still come from the profile, so the effective
        batch changes with the actual world size."""
        if world != self.nproc_per_node:
            log.warning(
                "compute profile expects %s with nproc_per_node=%d but "
                "WORLD_SIZE=%d — proceeding (effective batch scales with the "
                "actual world size)", self.launcher, self.nproc_per_node, world)


class ComputeConfig(_Frozen):
    target: str
    profiles: dict[str, ComputeProfile]

    @model_validator(mode="after")
    def _target_known(self) -> "ComputeConfig":
        if self.target not in self.profiles:
            raise ValueError(f"compute target {self.target!r} is not a profile: "
                             f"{sorted(self.profiles)}")
        return self


def load_compute_config(path: str | Path | None = None,
                        local_path: str | Path | None = None) -> ComputeConfig:
    """configs/compute.yaml (+ gitignored compute.local.yaml merged over it,
    unknown keys rejected — the paths.local.yaml pattern). An explicit `path`
    is taken verbatim with no local merge."""
    explicit = path is not None
    path = Path(path) if explicit else DEFAULT_COMPUTE_YAML
    with open(path, "r", encoding="utf-8") as f:
        raw: dict = yaml.safe_load(f) or {}
    local = (Path(local_path) if local_path is not None
             else (None if explicit else LOCAL_COMPUTE_YAML))
    if local is not None and local.exists():
        with open(local, "r", encoding="utf-8") as f:
            overrides: dict = yaml.safe_load(f) or {}
        unknown = set(overrides) - {"target", "profiles"}
        if unknown:
            raise KeyError(f"unknown keys in {local.name}: {sorted(unknown)}")
        if "profiles" in overrides:
            raw.setdefault("profiles", {}).update(overrides["profiles"])
        if "target" in overrides:
            raw["target"] = overrides["target"]
    return ComputeConfig.model_validate(raw)


def load_compute(selector: str | None = None) -> ComputeProfile:
    """selector: None -> configs/compute.yaml's target (+ local overrides);
    a profile name -> that profile from the default file; a path to a yaml ->
    that file's own target (no local merge)."""
    if selector and (selector.endswith((".yaml", ".yml")) or Path(selector).exists()):
        cfg = load_compute_config(selector)
        name = cfg.target
    else:
        cfg = load_compute_config()
        name = selector or cfg.target
        if name not in cfg.profiles:
            raise KeyError(f"unknown compute profile {name!r}; "
                           f"known: {sorted(cfg.profiles)}")
    profile = cfg.profiles[name]
    log.info("compute target: %s (%s)", name, profile.description or "-")
    return profile
