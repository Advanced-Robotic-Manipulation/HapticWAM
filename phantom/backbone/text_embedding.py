"""Text conditioning provider.

The released robot/action-cond checkpoint was post-trained with EMPTY-STRING
Cosmos-Reason1 embeddings (cr1_empty_string_text_embeddings.pt ships with the
weights); the live Reason1 encoder is not part of this download. We therefore
condition on the cached empty embedding. Per-episode language conditioning via
a live Reason1 encoder is an optional later upgrade (documented, not blocking).
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from phantom.config.backbone import BackboneConfig
from phantom.config.paths import PathsConfig

log = logging.getLogger(__name__)


class TextEmbeddingProvider:
    def __init__(self, bb: BackboneConfig, paths: PathsConfig | None, *,
                 fake: bool = False, device: str = "cpu",
                 dtype: torch.dtype = torch.float32):
        self.bb = bb
        self.device = device
        self.dtype = dtype
        if fake or paths is None:
            g = torch.Generator().manual_seed(1)
            self._emb = torch.randn(1, bb.text_emb_seq_len, bb.text_emb_dim,
                                    generator=g) * 0.02
        else:
            emb = torch.load(str(paths.cosmos_empty_text_embedding),
                             map_location="cpu", weights_only=True)
            if isinstance(emb, dict):
                emb = next(iter(emb.values()))
            assert emb.ndim == 3, f"unexpected text embedding shape {tuple(emb.shape)}"
            self._emb = emb
            log.info("cached empty-string text embedding: %s", tuple(emb.shape))
        self._emb = self._emb.to(device=device, dtype=dtype)

    def get(self, batch_size: int, text: list[str] | None = None) -> torch.Tensor:
        """(B, seq, dim). `text` is accepted for interface stability; with the
        cached-empty provider it is ignored (logged once)."""
        if text and any(t for t in text) and not getattr(self, "_warned", False):
            log.warning("per-episode text given but only the cached empty-string "
                        "embedding is available; ignoring text conditioning")
            self._warned = True
        return self._emb.expand(batch_size, -1, -1)
