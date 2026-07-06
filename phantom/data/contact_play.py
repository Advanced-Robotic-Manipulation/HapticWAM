"""ContactPlayDataset: full-resolution tactile keyframes for the SSL
contact-play pretrain (pipeline.md §7, train/pretrain_tactile.py).

Each item is one (8, H, W) float32 field stack (channels first), normalized
with the "fields" stats. Episodes are anything under data_root with recorded
keyframes — task-free contact play or regular demos alike.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from phantom.data.episode_store import EpisodeReader, list_episodes
from phantom.data.schema import NormStats, tactile_stream

log = logging.getLogger(__name__)


class ContactPlayDataset(Dataset):
    def __init__(self, data_root: Path, hw, norm: NormStats):
        self.hw = hw
        self.norm = norm
        self._readers: dict[Path, EpisodeReader] = {}
        self.index: list[tuple[Path, str, int]] = []
        for ep in list_episodes(Path(data_root)):
            r = EpisodeReader(ep)
            self._readers[ep] = r
            for s in hw.tactile.sensors:
                stream = tactile_stream(s.name, "keyframes")
                if r.has(stream):
                    self.index += [(ep, stream, i) for i in range(r.n(stream))]
        if not self.index:
            raise RuntimeError(f"no tactile keyframes under {data_root} — "
                               "record contact play or use --synthetic")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> torch.Tensor:
        ep, stream, k = self.index[i]
        frame = np.asarray(self._readers[ep]._g(stream)["data"][k], dtype=np.float32)
        frame = np.asarray(self.norm.normalize("fields", frame))
        return torch.from_numpy(np.ascontiguousarray(frame.transpose(2, 0, 1)))
