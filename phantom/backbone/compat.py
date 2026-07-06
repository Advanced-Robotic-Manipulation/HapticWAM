"""Compatibility shims for importing cosmos_predict2 on machines without
Linux-only CUDA packages (Windows dev box, and the 5090 box if Transformer
Engine is not built).

install() registers fake `transformer_engine` and `megatron` modules in
sys.modules BEFORE cosmos imports them. The shims cover exactly what the
import chain of the DiT + Wan VAE touches:

  - te.pytorch.RMSNorm                      (used by Attention / t_embedding_norm)
  - transformer_engine.pytorch.attention.rope.apply_rotary_pos_emb
  - transformer_engine.pytorch.attention.apply_rotary_pos_emb (fallback path)
  - transformer_engine.pytorch.attention.DotProductAttention  (raises if built —
      we force atten_backend="torch", so it must never be constructed)
  - megatron.core.parallel_state            (context-parallel introspection;
      returns "not initialized" so all CP paths stay disabled)

RISK (pipeline.md R6): the RMSNorm/RoPE shims must be numerically equivalent
to TE's kernels. scripts/verify_backbone.py runs a golden-parity check against
the repo's own inference on a Linux/TE machine — until that passes, treat
local full-checkpoint outputs as smoke-level only.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types

import torch
from torch import nn


def _make_module(name: str) -> types.ModuleType:
    """Module with a valid __spec__ (peft probes find_spec('transformer_engine')
    and raises on __spec__ = None)."""
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    return mod


# ---------------------------------------------------------------------------
# transformer_engine shims
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Numerics-matched stand-in for transformer_engine.pytorch.RMSNorm
    (zero_centered_gamma=False): y = x / sqrt(mean(x^2) + eps) * weight,
    computed in fp32 and cast back."""

    def __init__(self, hidden_size: int, eps: float = 1e-5, **kwargs):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x32 = x.float()
        y = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(dt)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rotary_pos_emb(t: torch.Tensor, freqs: torch.Tensor, *,
                         tensor_format: str = "sbhd", fused: bool = False,
                         **kwargs) -> torch.Tensor:
    """TE-compatible RoPE for tensor_format='bshd': t (B, S, H, D),
    freqs (S, 1, 1, D) holding angles (duplicated-halves layout)."""
    assert tensor_format == "bshd", f"shim only supports bshd, got {tensor_format}"
    S = t.shape[1]
    angles = freqs[:S].to(torch.float32)             # (S, 1, 1, D)
    cos = torch.cos(angles).transpose(0, 1)          # (1, S, 1, D) broadcast over B, H
    sin = torch.sin(angles).transpose(0, 1)
    t32 = t.to(torch.float32)
    out = t32 * cos + _rotate_half(t32) * sin
    return out.to(t.dtype)


class DotProductAttention(nn.Module):  # pragma: no cover - must never be built
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "transformer_engine shim: DotProductAttention requested — the model was "
            "built with atten_backend='transformer_engine'. PHANTOM requires "
            "atten_backend='torch' (BackboneConfig forces it); check the net kwargs.")


def _build_te_module() -> types.ModuleType:
    te = _make_module("transformer_engine")
    te_pytorch = _make_module("transformer_engine.pytorch")
    te_attention = _make_module("transformer_engine.pytorch.attention")
    te_rope = _make_module("transformer_engine.pytorch.attention.rope")

    te_pytorch.RMSNorm = RMSNorm
    # peft's LoRA dispatcher isinstance-checks these TE layer types; distinct
    # never-instantiated placeholder classes make those checks simply False.
    te_pytorch.LayerNormLinear = type("LayerNormLinear", (), {})
    te_pytorch.LayerNormMLP = type("LayerNormMLP", (), {})
    te_pytorch.Linear = type("TELinear", (), {})
    te_rope.apply_rotary_pos_emb = apply_rotary_pos_emb
    te_attention.apply_rotary_pos_emb = apply_rotary_pos_emb
    te_attention.DotProductAttention = DotProductAttention
    te_attention.rope = te_rope
    te_pytorch.attention = te_attention
    te.pytorch = te_pytorch
    te.__version__ = "0.0.0-phantom-shim"

    sys.modules["transformer_engine"] = te
    sys.modules["transformer_engine.pytorch"] = te_pytorch
    sys.modules["transformer_engine.pytorch.attention"] = te_attention
    sys.modules["transformer_engine.pytorch.attention.rope"] = te_rope
    return te


# ---------------------------------------------------------------------------
# megatron shim (wan2pt1.py hard-imports megatron.core.parallel_state)
# ---------------------------------------------------------------------------

def _build_megatron_module() -> types.ModuleType:
    mg = _make_module("megatron")
    core = _make_module("megatron.core")
    ps = _make_module("megatron.core.parallel_state")

    ps.is_initialized = lambda: False
    ps.get_context_parallel_group = lambda: (_ for _ in ()).throw(
        RuntimeError("megatron shim: context parallel is not available"))
    ps.get_context_parallel_world_size = lambda: 1
    ps.get_context_parallel_rank = lambda: 0

    core.parallel_state = ps
    mg.core = core
    sys.modules["megatron"] = mg
    sys.modules["megatron.core"] = core
    sys.modules["megatron.core.parallel_state"] = ps
    return mg


# ---------------------------------------------------------------------------
# multistorageclient shim (easy_io's MSC backend hard-imports it; the pip
# package needs Linux-only xattr. Only the class NAMES are needed at import
# time — constructing them means someone actually configured an MSC store.)
# ---------------------------------------------------------------------------

def _build_msc_module() -> types.ModuleType:
    msc = _make_module("multistorageclient")
    msc_types = _make_module("multistorageclient.types")

    class _Unavailable:
        def __init__(self, *a, **k):
            raise RuntimeError("multistorageclient shim: MSC storage is not "
                               "available in this environment")

    msc.StorageClient = _Unavailable
    msc.StorageClientConfig = _Unavailable
    msc_types.Range = _Unavailable
    msc.types = msc_types
    msc.__version__ = "0.0.0-phantom-shim"
    sys.modules["multistorageclient"] = msc
    sys.modules["multistorageclient.types"] = msc_types
    return msc


# ---------------------------------------------------------------------------

_installed = False


def install(force: bool = False) -> dict[str, bool]:
    """Idempotent. Installs a shim only where the real package is missing.
    Returns {package: shimmed?} for logging."""
    global _installed
    result = {}
    for pkg, builder in (("transformer_engine", _build_te_module),
                         ("megatron", _build_megatron_module),
                         ("multistorageclient", _build_msc_module)):
        if pkg in sys.modules and not force:
            result[pkg] = getattr(sys.modules[pkg], "__version__", "") == "0.0.0-phantom-shim"
            continue
        try:
            if force:
                raise ImportError
            __import__(pkg)
            result[pkg] = False          # real package present
        except ImportError:
            builder()
            result[pkg] = True
    _installed = True
    return result


def is_shimmed() -> bool:
    mod = sys.modules.get("transformer_engine")
    return mod is not None and getattr(mod, "__version__", "") == "0.0.0-phantom-shim"
