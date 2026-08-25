"""Precompute Cosmos-Reason1 text embeddings for task/instruction strings.

The released robot/action-cond base was post-trained with Cosmos-Reason1
embeddings; only the EMPTY-string embedding ships with the weights, so text
conditioning has been inert (TextEmbeddingProvider returns the cached empty
embedding for every sample). This script closes that gap OFFLINE: it embeds
each distinct instruction string once and writes a cache that
TextEmbeddingProvider loads at train/deploy time. No live encoder is ever
needed in the training loop.

Recipe (replicated from cosmos_predict2 text_encoders/text_encoder.py,
EmbeddingConcatStrategy.FULL_CONCAT):
  chat template [system: "You are a helpful assistant who will provide
  prompts to an image generator." / user: <text>] -> pad or truncate to 512
  tokens -> QwenVL forward -> hidden layers 1..N, each mean-normalized over
  the feature dim -> concat over features -> (512, n_layers*hidden).

Weights: nvidia/Cosmos-Reason1-7B (ungated on HF). The public release is not
byte-identical to the internal SFT iteration that produced the shipped empty
embedding, so ALWAYS run --verify first: it embeds "" and reports similarity
against cr1_empty_string_text_embeddings.pt. High cosine (> 0.99) means the
recipe+weights reproduce the base model's conditioning space; low similarity
means STOP and investigate before training with the cache.

Usage:
  # 1. verify the recipe reproduces the shipped empty-string embedding
  python -m phantom.scripts.embed_task_texts --verify

  # 2. embed every distinct text in a dataset (reads manifests/episode metas)
  python -m phantom.scripts.embed_task_texts --data <root>/tasks \
      --out <root>/tasks/text_embeddings.pt

  # 3. or embed explicit strings
  python -m phantom.scripts.embed_task_texts --texts "wipe the whiteboard" \
      "pick up the egg" --out text_embeddings.pt

The cache is a dict {text: (512, D) bf16 tensor} plus a "__meta__" entry
recording model id + recipe, so provenance survives into checkpoints.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from phantom.config.paths import load_paths

log = logging.getLogger(__name__)

MODEL_ID = "nvidia/Cosmos-Reason1-7B"
SYSTEM_PROMPT = ("You are a helpful assistant who will provide prompts to an "
                 "image generator.")
NUM_TOKENS = 512


def collect_texts(data_root: Path) -> list[str]:
    """Distinct instruction strings for a dataset root: manifest text/task
    fields when present, else episode meta.json text/task."""
    texts: set[str] = set()
    man = data_root.parent / "manifests" / "all.jsonl"
    if man.exists():
        for line in man.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                texts.add(str(r.get("text") or r.get("task") or ""))
    else:
        for mp in sorted(data_root.rglob("ep_*/meta.json")):   # follows symlinked eps
            try:
                m = json.loads(mp.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            texts.add(str(m.get("text") or m.get("task") or ""))
    texts.discard("")
    return sorted(texts)


class Reason1Embedder:
    """Text-only Reason1 forward implementing the FULL_CONCAT recipe."""

    def __init__(self, device: str = "cuda", model_id: str = MODEL_ID):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.device = device
        log.info("loading %s on %s (bf16)...", model_id, device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=torch.bfloat16,
            output_hidden_states=True).to(device).eval()
        tok = self.processor.tokenizer
        self.pad_id = tok.pad_token_id
        assert self.pad_id is not None, "tokenizer has no pad token"

    def _input_ids(self, text: str) -> torch.Tensor:
        conv = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [{"type": "text", "text": text}]}]
        ids = self.processor.apply_chat_template(
            conv, tokenize=True, add_generation_prompt=False)
        ids = list(ids[0] if isinstance(ids[0], (list, tuple)) else ids)
        if len(ids) < NUM_TOKENS:
            ids = ids + [self.pad_id] * (NUM_TOKENS - len(ids))
        else:
            ids = ids[:NUM_TOKENS]
        return torch.tensor(ids, dtype=torch.long, device=self.device)[None]

    @staticmethod
    def _mean_normalize(t: torch.Tensor) -> torch.Tensor:
        return (t - t.mean(dim=-1, keepdim=True)) / (t.std(dim=-1, keepdim=True) + 1e-8)

    @torch.no_grad()
    def embed(self, text: str) -> torch.Tensor:
        """(NUM_TOKENS, n_layers*hidden) fp32 on cpu."""
        out = self.model(input_ids=self._input_ids(text),
                         output_hidden_states=True)
        hs = out.hidden_states                     # (n_layers+1) x (1, T, H)
        parts = [self._mean_normalize(h[0].float()) for h in hs[1:]]
        return torch.cat(parts, dim=-1).cpu()


def verify(embedder: Reason1Embedder, paths) -> bool:
    ref = torch.load(str(paths.cosmos_empty_text_embedding),
                     map_location="cpu", weights_only=True)
    if isinstance(ref, dict):
        ref = next(iter(ref.values()))
    ref = ref[0].float()                           # (512, D)
    ours = embedder.embed("")
    if ours.shape != ref.shape:
        log.error("SHAPE MISMATCH ours=%s ref=%s", tuple(ours.shape), tuple(ref.shape))
        return False
    cos = torch.nn.functional.cosine_similarity(ours, ref, dim=-1)
    log.info("empty-string embedding vs shipped: cosine mean=%.4f min=%.4f "
             "| max|diff|=%.4f", cos.mean(), cos.min(), (ours - ref).abs().max())
    ok = bool(cos.mean() > 0.99)
    log.info("VERIFY %s", "PASSED" if ok else
             "FAILED — do NOT train with this cache; the public Reason1-7B "
             "does not reproduce the base model's conditioning space")
    return ok


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="", help="dataset root (…/tasks); embeds its distinct texts")
    ap.add_argument("--texts", nargs="*", default=None, help="explicit strings to embed")
    ap.add_argument("--out", default="", help="output cache path (.pt)")
    ap.add_argument("--verify", action="store_true",
                    help="embed \"\" and compare against the shipped empty embedding")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--paths", default=None)
    args = ap.parse_args()

    paths = load_paths(args.paths)
    emb = Reason1Embedder(device=args.device, model_id=args.model)

    if args.verify and not verify(emb, paths):
        return 1

    texts = list(args.texts or [])
    if args.data:
        texts += collect_texts(Path(args.data))
    texts = sorted(set(t for t in texts if t))
    if not texts:
        log.info("no texts requested — done (verify-only run)")
        return 0

    out = Path(args.out or "text_embeddings.pt")
    cache: dict = {}
    if out.exists():                               # incremental: keep old entries
        cache = torch.load(str(out), map_location="cpu", weights_only=True)
        log.info("extending existing cache (%d entries)", len(cache) - ("__meta__" in cache))
    for i, t in enumerate(texts, 1):
        if t in cache:
            log.info("[%d/%d] cached: %r", i, len(texts), t)
            continue
        log.info("[%d/%d] embedding %r", i, len(texts), t)
        cache[t] = emb.embed(t).to(torch.bfloat16)
    cache["__meta__"] = {"model": args.model, "recipe": "reason1_full_concat_v1",
                         "num_tokens": NUM_TOKENS, "system_prompt": SYSTEM_PROMPT}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    torch.save(cache, tmp)
    tmp.replace(out)
    log.info("wrote %s (%d texts)", out, len(cache) - 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
