"""Paths configuration: cosmos repo/weights locations + data/run roots.

configs/paths.yaml holds shared defaults; configs/paths.local.yaml (gitignored)
is merged over it for per-machine overrides.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
DEFAULT_PATHS_YAML = CONFIGS_DIR / "paths.yaml"
LOCAL_PATHS_YAML = CONFIGS_DIR / "paths.local.yaml"


def _root(value: str | Path) -> Path:
    """Resolve a paths.yaml entry: relative entries are REPO-root-relative (so
    the committed defaults work from any working directory and on any machine),
    absolute entries — what paths.local.yaml normally carries — are used as-is."""
    p = Path(value)
    return p if p.is_absolute() else REPO_ROOT / p


@dataclass(frozen=True)
class PathsConfig:
    cosmos_repo: Path
    cosmos_weights_root: Path
    cosmos_checkpoint: Path            # absolute (resolved against weights_root)
    cosmos_empty_text_embedding: Path  # absolute
    cosmos_tokenizer: Path             # absolute
    data_root: Path
    runs_root: Path
    # optional per-text embedding cache (embed_task_texts.py output); empty ->
    # text conditioning stays on the cached empty-string embedding
    cosmos_text_embedding_cache: str = ""

    def episodes_root(self) -> Path:
        return self.data_root / "episodes"

    def contact_play_root(self) -> Path:
        return self.data_root / "contact_play"

    def validate(self, *, require_cosmos: bool = False) -> None:
        """Existence checks. Cosmos paths are warn-only unless require_cosmos
        (the Windows dev box may not need them); data_root parent must exist."""
        for name in ("cosmos_repo", "cosmos_weights_root", "cosmos_checkpoint",
                     "cosmos_empty_text_embedding", "cosmos_tokenizer"):
            p: Path = getattr(self, name)
            if not p.exists():
                msg = f"paths.{name} does not exist: {p}"
                if require_cosmos:
                    raise FileNotFoundError(msg)
                log.warning(msg)
        if not self.data_root.parent.exists():
            raise FileNotFoundError(f"parent of data_root does not exist: {self.data_root.parent}")


def load_paths(path: str | Path | None = None,
               local_path: str | Path | None = None) -> PathsConfig:
    path = Path(path) if path is not None else DEFAULT_PATHS_YAML
    with open(path, "r", encoding="utf-8") as f:
        raw: dict = yaml.safe_load(f) or {}

    local = Path(local_path) if local_path is not None else LOCAL_PATHS_YAML
    if local.exists():
        with open(local, "r", encoding="utf-8") as f:
            overrides = yaml.safe_load(f) or {}
        unknown = set(overrides) - set(raw)
        if unknown:
            raise KeyError(f"unknown keys in {local.name}: {sorted(unknown)}")
        raw.update(overrides)

    weights_root = _root(raw["cosmos_weights_root"])
    return PathsConfig(
        cosmos_repo=_root(raw["cosmos_repo"]),
        cosmos_weights_root=weights_root,
        cosmos_checkpoint=weights_root / raw["cosmos_checkpoint"],
        cosmos_empty_text_embedding=weights_root / raw["cosmos_empty_text_embedding"],
        cosmos_text_embedding_cache=str(raw.get("cosmos_text_embedding_cache", "") or ""),
        cosmos_tokenizer=weights_root / raw["cosmos_tokenizer"],
        data_root=_root(raw["data_root"]),
        runs_root=_root(raw["runs_root"]),
    )
