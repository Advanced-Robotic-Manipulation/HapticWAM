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


def install_bias_hooks(blocks, holder: BiasHolder) -> None:
    """Wrap every block's SELF-attention op (cross-attention untouched)."""
    for i, block in enumerate(blocks):
        attn = block.self_attn
        assert attn.backend == "torch", (
            f"attention-bias hook requires atten_backend='torch', block {i} uses "
            f"{attn.backend!r}")
        attn.attn_op = BiasedSDPAOp(holder, i)


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
