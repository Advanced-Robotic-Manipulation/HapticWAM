"""Backbone loader: makes cosmos_predict2 importable from the configured repo
path (after compat shims), builds the PhantomDiT, loads the base checkpoint,
and injects LoRA.

Key-remapping strategy (pipeline plan §8): every PHANTOM-new parameter name
starts with "phantom_", so base/new separation on load is a single prefix
test. LoRA is injected AFTER the base load so base keys never need
`base_layer.` remapping.
"""

from __future__ import annotations

import hashlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from phantom.backbone import compat
from phantom.config.backbone import BackboneConfig
from phantom.config.hardware import HardwareConfig
from phantom.config.model import PhantomModelConfig
from phantom.config.paths import PathsConfig
from phantom.model.sequence import SequenceLayout

log = logging.getLogger(__name__)

_cosmos_ready = False


def setup_cosmos(paths: PathsConfig) -> None:
    """Install shims + put the cosmos repo on sys.path. Idempotent."""
    global _cosmos_ready
    if _cosmos_ready:
        return
    shims = compat.install()
    repo = str(paths.cosmos_repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    _shim_cosmos_cuda(paths)
    try:
        import cosmos_predict2  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            f"cannot import cosmos_predict2 from {repo} — check paths.yaml "
            f"(cosmos_repo) and the repo clone") from e
    log.info("cosmos_predict2 importable from %s (shims: %s)", repo, shims)
    _cosmos_ready = True


# ---------------------------------------------------------------------------

def build_phantom_net(bb: BackboneConfig, mc: PhantomModelConfig, hw: HardwareConfig,
                      layout: SequenceLayout, paths: PathsConfig,
                      device: str = "cpu", dtype: torch.dtype = torch.float32):
    """Construct a PhantomDiT with the transcribed backbone kwargs.

    NOTE on max sizes: max_frames bounds the RoPE/temporal axis — must cover
    the EXTENDED layout, not just the video window. Larger max_* only sizes
    position tables; it does not change weights.
    """
    assert bb.atten_backend == "torch", (
        "PHANTOM requires atten_backend='torch' (SDPA): the ACC attention-bias hook "
        "wraps torch_attention_op's attn_mask argument")
    setup_cosmos_from_default(paths)
    from phantom.model.phantom_dit import PhantomDiT

    net = PhantomDiT(
        layout=layout, mc=mc, bb=bb, hw=hw,
        # ---- cosmos MiniTrainDIT kwargs (transcribed; see BackboneConfig) ----
        max_img_h=bb.res_h, max_img_w=bb.res_w,
        max_frames=max(layout.t_total, bb.t_video) * bb.patch_temporal,
        in_channels=bb.in_channels, out_channels=bb.out_channels,
        patch_spatial=bb.patch_spatial, patch_temporal=bb.patch_temporal,
        model_channels=bb.model_channels, num_blocks=bb.num_blocks,
        num_heads=bb.num_heads, atten_backend=bb.atten_backend,
        crossattn_emb_channels=bb.crossattn_emb_channels,
        use_crossattn_projection=bb.use_crossattn_projection,
        crossattn_proj_in_channels=bb.crossattn_proj_in_channels,
        pos_emb_cls=bb.pos_emb_cls, pos_emb_learnable=bb.pos_emb_learnable,
        use_adaln_lora=bb.use_adaln_lora, adaln_lora_dim=bb.adaln_lora_dim,
        rope_h_extrapolation_ratio=bb.rope_h_extrapolation_ratio,
        rope_w_extrapolation_ratio=bb.rope_w_extrapolation_ratio,
        rope_t_extrapolation_ratio=bb.rope_t_extrapolation_ratio,
        use_wan_fp32_strategy=bb.use_wan_fp32_strategy,
        timestep_scale=bb.timestep_scale,
        action_dim=bb.action_dim,
        temporal_compression_ratio=bb.actions_per_latent_frame,
    )
    return net.to(device=device, dtype=dtype)


def setup_cosmos_from_default(paths: PathsConfig) -> None:
    setup_cosmos(paths)


def _shim_cosmos_cuda(paths: PathsConfig) -> None:
    """cosmos_predict2/__init__ requires a `cosmos_cuda` companion package as
    a version marker for the 'uv sync --extra=cuXXX' install. We only import
    model classes as a library, so a version-matched stub suffices when the
    real extra is not installed (Windows dev box / non-uv envs)."""
    try:
        import cosmos_cuda  # noqa: F401
        return
    except ImportError:
        pass
    import re
    import types
    about = paths.cosmos_repo / "cosmos_predict2" / "__about__.py"
    m = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]",
                  about.read_text(encoding="utf-8"))
    stub = types.ModuleType("cosmos_cuda")
    stub.__version__ = m.group(1) if m else "0.0.0"
    sys.modules["cosmos_cuda"] = stub


# ---------------------------------------------------------------------------

@dataclass
class LoadReport:
    n_loaded: int
    n_dropped: int
    missing_phantom: int
    sha256: str


def load_base_weights(net, ckpt_path: Path, *, verify: BackboneConfig | None = None,
                      compute_sha: bool = False) -> LoadReport:
    """Load the released robot/action-cond `.pt` (EMA bf16) into a PhantomDiT.

    - strips the 'net.' prefix
    - drops 'net.accum_*' counters and TE '_extra_state' blobs
    - strict=False; asserts: no unexpected keys, every missing key is phantom_*
    """
    sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=True, mmap=True)
    if verify is not None:
        verify.verify_against_state_dict(sd)
    dropped = 0
    clean = {}
    for k, v in sd.items():
        if not k.startswith("net."):
            dropped += 1
            continue
        k2 = k[len("net."):]
        if k2.startswith("accum_") or "_extra_state" in k2:
            dropped += 1
            continue
        clean[k2] = v
    missing, unexpected = net.load_state_dict(clean, strict=False)
    bad_unexpected = [k for k in unexpected]
    assert not bad_unexpected, f"checkpoint keys not accepted by the net: {bad_unexpected[:10]}"
    non_phantom_missing = [k for k in missing if not k.startswith("phantom_")]
    assert not non_phantom_missing, (
        f"non-phantom parameters missing from checkpoint (naming rule violated or "
        f"config drift): {non_phantom_missing[:10]}")
    sha = ""
    if compute_sha:
        sha = hashlib.sha256(Path(ckpt_path).read_bytes()).hexdigest()
    log.info("base checkpoint loaded: %d tensors, %d dropped, %d phantom params fresh",
             len(clean), dropped, len(missing))
    return LoadReport(n_loaded=len(clean), n_dropped=dropped,
                      missing_phantom=len(missing), sha256=sha)


# ---------------------------------------------------------------------------

def inject_lora(net, mc: PhantomModelConfig) -> int:
    """Mirror cosmos's add_lora_to_net (text2world_model.py) via peft, applied
    AFTER base load. Returns the number of LoRA parameters."""
    from peft import LoraConfig, inject_adapter_in_model

    lora_config = LoraConfig(
        r=mc.lora.rank,
        lora_alpha=mc.lora.alpha,
        init_lora_weights=True,
        target_modules=mc.lora.target_modules.split(","),
        lora_dropout=mc.lora.dropout,
    )
    inject_adapter_in_model(lora_config, net)
    n = sum(p.numel() for name, p in net.named_parameters() if "lora_" in name)
    log.info("LoRA injected: r=%d alpha=%d targets=%s (%.1f M params)",
             mc.lora.rank, mc.lora.alpha, mc.lora.target_modules, n / 1e6)
    return n


def set_trainable(net, *, train_lora: bool = True, train_phantom: bool = True) -> dict:
    """Freeze everything except lora_* and phantom_* parameters."""
    counts = {"lora": 0, "phantom": 0, "frozen": 0}
    for name, p in net.named_parameters():
        if "lora_" in name:
            p.requires_grad = train_lora
            counts["lora"] += p.numel()
        elif name.startswith("phantom_") or ".phantom_" in name:
            p.requires_grad = train_phantom
            counts["phantom"] += p.numel()
        else:
            p.requires_grad = False
            counts["frozen"] += p.numel()
    log.info("trainable: lora %.1fM, phantom %.1fM, frozen %.1fM",
             counts["lora"] / 1e6, counts["phantom"] / 1e6, counts["frozen"] / 1e6)
    return counts
