"""Additive attention-bias hook for the cosmos DiT (ACC injection + the
ACE structural droppable-video mask).

Mechanism: cosmos's Attention with atten_backend="torch" sets
`self.attn_op = torch_attention_op`, and `compute_attention` calls
`self.attn_op(q, k, v)` with q,k,v (B, S, H, D). torch_attention_op accepts an
`attn_mask` kwarg (additive float mask for SDPA) that compute_attention never
passes — so wrapping attn_op per block is a weight-free, fork-free hook.

The bias tensor lives in a BiasHolder that PhantomDiT fills before the block
loop and clears after (try/finally); each block applies its own learnable
scale to the ACC component (per-block lambda, init 0 => exact no-op at step 0).
"""

from __future__ import annotations

import numpy as np
import torch

from phantom.model.sequence import SequenceLayout


class BiasHolder:
    """Mutable slot shared by all wrapped attn ops. NOT an nn.Module — holds
    no parameters, only the per-forward bias tensors."""

    def __init__(self):
        self.structural: torch.Tensor | None = None   # (1, 1, S, S) additive {0, -1e4}
        self.acc_key: torch.Tensor | None = None      # (B, 1, 1, S) ACC key bias (unscaled)
        self.lambdas: list[torch.Tensor] | None = None  # per-block learnable scales
        # flex-attention caches (persist across forwards; see FlexBiasedOp)
        self._flex_masks: dict = {}       # (id(structural), S_pad, dev) -> BlockMask
        self._flex_kv: tuple | None = None  # (id(acc_key), padded (B, S_pad) fp32)

    def flex_block_mask(self, S: int, S_pad: int, device) -> "object":
        """BlockMask equivalent of the structural additive mask; cached per
        structural-tensor identity (PhantomDiT caches those per layout)."""
        key = (id(self.structural), S_pad, str(device))
        bm = self._flex_masks.get(key)
        if bm is None:
            from torch.nn.attention.flex_attention import create_block_mask
            allowed = torch.zeros(S_pad, S_pad, dtype=torch.bool, device=device)
            allowed[:S, :S] = self.structural[0, 0] == 0

            def mask_mod(b, h, q_idx, kv_idx):
                return allowed[q_idx, kv_idx]

            bm = create_block_mask(mask_mod, B=None, H=None,
                                   Q_LEN=S_pad, KV_LEN=S_pad, device=device)
            self._flex_masks[key] = bm
        return bm

    def flex_kv_bias(self, S_pad: int) -> torch.Tensor:
        """(B, S_pad) fp32 view of acc_key, zero-padded; cached per forward."""
        ak = self.acc_key
        if self._flex_kv is None or self._flex_kv[0] != id(ak):
            v = ak.reshape(ak.shape[0], -1).float()
            if v.shape[1] < S_pad:
                v = torch.nn.functional.pad(v, (0, S_pad - v.shape[1]))
            self._flex_kv = (id(ak), v)
        return self._flex_kv[1]

    def bias_for_block(self, block_idx: int) -> torch.Tensor | None:
        parts = []
        if self.structural is not None:
            parts.append(self.structural)
        if self.acc_key is not None and self.lambdas is not None:
            parts.append(self.lambdas[block_idx] * self.acc_key)
        if not parts:
            return None
        out = parts[0]
        for p in parts[1:]:
            out = out + p
        return out

    def clear(self) -> None:
        self.structural = None
        self.acc_key = None
        self.lambdas = None


class BiasedSDPAOp:
    """Drop-in replacement for torch_attention_op on one block's self_attn."""

    def __init__(self, holder: BiasHolder, block_idx: int):
        self.holder = holder
        self.block_idx = block_idx

    def __call__(self, q, k, v, **kwargs):
        from cosmos_predict2._src.predict2.networks.minimal_v4_dit import torch_attention_op
        bias = self.holder.bias_for_block(self.block_idx)
        if bias is not None:
            bias = bias.to(dtype=q.dtype, device=q.device)
        return torch_attention_op(q, k, v, attn_mask=bias, **kwargs)

    # cosmos calls this on attn ops when wiring context parallel
    def set_context_parallel_group(self, *args, **kwargs) -> None:
        return None


class FlexBiasedOp:
    """FlexAttention drop-in for one block's self_attn (inference).

    Same math as BiasedSDPAOp, faster kernels: the structural {0, -1e4}
    additive mask becomes a real BlockMask (forbidden tiles are SKIPPED, not
    re-scored — CONTACT/ACTION x VIDEO_GEN tiles never run), and the ACC key
    bias becomes a score_mod `score + lambda_i * acc_key[b, kv]`. Sequences
    are right-padded to a 128 multiple (flex kernel granularity); padded rows
    are masked out and sliced off."""

    _flex = None    # class-shared torch.compile'd flex_attention

    def __init__(self, holder: BiasHolder, block_idx: int):
        self.holder = holder
        self.block_idx = block_idx

    @classmethod
    def _get_flex(cls):
        if cls._flex is None:
            from torch.nn.attention.flex_attention import flex_attention
            cls._flex = torch.compile(flex_attention, dynamic=False)
        return cls._flex

    @torch._dynamo.disable
    def __call__(self, q, k, v, flatten_heads: bool = True, **kwargs):
        # dynamo.disable: if the caller block is itself torch.compile'd, a
        # nested compile is IGNORED and flex silently runs its eager math
        # path (materializes the S x S scores; measured 5.6x slower). The
        # graph-break here lets the block compile around us while this op
        # keeps its own compiled flex kernel.
        from einops import rearrange
        holder = self.holder
        B, S, _H, _D = q.shape
        s_pad = (S + 127) // 128 * 128
        qh = rearrange(q, "b s h d -> b h s d")
        kh = rearrange(k, "b s h d -> b h s d")
        vh = rearrange(v, "b s h d -> b h s d")
        if s_pad != S:
            pad = (0, 0, 0, s_pad - S)                 # pad the S dim
            qh = torch.nn.functional.pad(qh, pad)
            kh = torch.nn.functional.pad(kh, pad)
            vh = torch.nn.functional.pad(vh, pad)

        block_mask = holder.flex_block_mask(S, s_pad, q.device)
        score_mod = None
        if holder.acc_key is not None and holder.lambdas is not None:
            lam = holder.lambdas[self.block_idx]
            kv_bias = holder.flex_kv_bias(s_pad)       # (B, S_pad) fp32

            def score_mod(score, b, h, q_idx, kv_idx):
                return score + lam * kv_bias[b, kv_idx]

        out = self._get_flex()(qh, kh, vh, score_mod=score_mod,
                               block_mask=block_mask)
        out = out[:, :, :S]
        if flatten_heads:
            return rearrange(out, "b h s d -> b s (h d)")
        return rearrange(out, "b h s d -> b s h d")

    def set_context_parallel_group(self, *args, **kwargs) -> None:
        return None


def install_bias_hooks(blocks, holder: BiasHolder) -> None:
    """Wrap every block's SELF-attention op (cross-attention untouched)."""
    for i, block in enumerate(blocks):
        attn = block.self_attn
        assert attn.backend == "torch", (
            f"attention-bias hook requires atten_backend='torch', block {i} uses "
            f"{attn.backend!r}")
        attn.attn_op = BiasedSDPAOp(holder, i)


def install_flex_hooks(blocks, holder: BiasHolder) -> None:
    """Swap every block's self_attn op to the FlexAttention variant
    (inference-only; call after install_bias_hooks/weight load)."""
    for i, block in enumerate(blocks):
        block.self_attn.attn_op = FlexBiasedOp(holder, i)


# ---------------------------------------------------------------------------
# bias construction from the layout
# ---------------------------------------------------------------------------

def structural_bias_tokens(layout: SequenceLayout, device="cpu") -> torch.Tensor:
    """(1, 1, S, S) token-level expansion of the frame-level structural mask
    (CONTACT/ACTION queries never attend VIDEO_GEN keys)."""
    frame_bias = torch.from_numpy(layout.structural_attn_bias())      # (T, T)
    n = layout.tokens_per_frame
    tok = frame_bias.repeat_interleave(n, dim=0).repeat_interleave(n, dim=1)
    return tok.unsqueeze(0).unsqueeze(0).to(device)


def acc_key_bias(layout: SequenceLayout, gate_B: torch.Tensor,
                 beta_evt_B: torch.Tensor) -> torch.Tensor:
    """(B, 1, 1, S): key-only additive bias `beta(p_evt) * g` on the haptic
    group's tokens (teacher: OBS_MECH+OBS_GEL+CONTACT; student:
    CONTACT+OBS_PROPRIO). Per-block scaling is applied in BiasHolder."""
    mask = torch.from_numpy(layout.haptic_group_token_mask()).to(gate_B.device)
    val = (beta_evt_B * gate_B).reshape(-1, 1, 1, 1)          # (B,1,1,1)
    return val * mask.reshape(1, 1, 1, -1).to(val.dtype)
