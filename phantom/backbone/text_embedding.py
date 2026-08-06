"""Text conditioning provider.

The released robot/action-cond checkpoint was post-trained with Cosmos-Reason1
embeddings; only the EMPTY-string embedding ships with the weights
(cr1_empty_string_text_embeddings.pt), so the default is to condition on that
cached embedding for every sample — language conditioning inert (pipeline.md
lists a per-episode instruction `l` as an input, so this is a gap, not a
design choice).

PER-EPISODE TEXT CONDITIONING: precompute embeddings once with
`phantom.scripts.embed_task_texts` (run its --verify gate first!) and point
`paths.cosmos_text_embedding_cache` at the resulting .pt. The provider then
looks each batch text up in the cache, falling back to the empty embedding
for unknown/empty strings. No live encoder ever runs in the training loop,
and with no cache configured behavior is bit-identical to before.
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
        # optional per-text cache (embed_task_texts.py output)
        self._cache: dict[str, torch.Tensor] = {}
        self._unknown_warned: set[str] = set()
        cache_path = getattr(paths, "cosmos_text_embedding_cache", "") if paths else ""
        if cache_path and Path(cache_path).exists():
            raw = torch.load(str(cache_path), map_location="cpu", weights_only=True)
            meta = raw.pop("__meta__", {})
            for k, v in raw.items():
                assert v.shape == self._emb.shape[1:], (
                    f"text cache entry {k!r} shape {tuple(v.shape)} != "
                    f"{tuple(self._emb.shape[1:])}")
                self._cache[k] = v.to(device=device, dtype=dtype)
            log.info("text embedding cache: %d entries from %s (recipe=%s)",
                     len(self._cache), cache_path, meta.get("recipe", "?"))
        elif cache_path:
            log.warning("cosmos_text_embedding_cache=%s does not exist — "
                        "text conditioning stays inert", cache_path)

    def get(self, batch_size: int, text: list[str] | None = None) -> torch.Tensor:
        """(B, seq, dim). With a cache configured, each sample's text is looked
        up individually (empty/unknown -> the empty-string embedding). Without
        a cache: the historical behavior, empty embedding for every sample."""
        if self._cache and text:
            rows = []
            for t in list(text)[:batch_size]:
                e = self._cache.get(t or "")
                if e is None:
                    if t and t not in self._unknown_warned:
                        self._unknown_warned.add(t)
                        log.warning("text %r not in the embedding cache — using "
                                    "the empty-string embedding (add it via "
                                    "embed_task_texts.py)", t)
                    e = self._emb[0]
                rows.append(e)
            while len(rows) < batch_size:
                rows.append(self._emb[0])
            return torch.stack(rows)
        if text and any(t for t in text) and not getattr(self, "_warned", False):
            log.warning("per-episode text given but only the cached empty-string "
                        "embedding is available; ignoring text conditioning")
            self._warned = True
        return self._emb.expand(batch_size, -1, -1)
