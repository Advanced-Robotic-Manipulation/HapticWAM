"""Model factory shared by all training/inference entry points.

build_model() assembles: SequenceLayout -> PhantomDiT (base weights + LoRA,
or tiny random preset) -> HHT -> PhantomRectifiedFlow. The `tiny` flag swaps
the 2B backbone for BackboneConfig.tiny() with a FakeVAE and fake text
embeddings — CPU-runnable end to end.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from phantom.backbone import loader as bl
from phantom.backbone.text_embedding import TextEmbeddingProvider
from phantom.backbone.vae import make_vae
from phantom.config.backbone import BackboneConfig
from phantom.config.hardware import HardwareConfig
from phantom.config.model import PhantomModelConfig
from phantom.config.paths import PathsConfig
from phantom.model.sequence import SequenceLayout

log = logging.getLogger(__name__)


@dataclass
class PhantomModel:
    rf: "object"                     # PhantomRectifiedFlow
    layout: SequenceLayout
    bb: BackboneConfig
    mc: PhantomModelConfig
    hw: HardwareConfig


def build_model(hw: HardwareConfig, paths: PathsConfig, *,
                mc: PhantomModelConfig | None = None,
                student: bool = False, tiny: bool = False,
                load_base: bool = True, lora: bool = True,
                device: str = "cpu",
                dtype: torch.dtype = torch.float32) -> PhantomModel:
    """tiny=True still imports the cosmos repo (for the DiT classes) but uses
    the small preset, random init, FakeVAE and fake text — CPU-runnable."""
    if mc is None:
        mc = PhantomModelConfig(student=student)
    assert mc.student == student, "mc.student must agree with the student flag"
    bb = BackboneConfig.tiny() if tiny else BackboneConfig()
    layout = SequenceLayout.build(bb, mc, hw, student=student)
    log.info("layout:\n%s", layout.describe())

    bl.setup_cosmos(paths)
    net = bl.build_phantom_net(bb, mc, hw, layout, paths, device=device, dtype=dtype)
    if load_base and not tiny:
        bl.load_base_weights(net, paths.cosmos_checkpoint, verify=bb)
    if lora:
        bl.inject_lora(net, mc)
    bl.set_trainable(net)

    vae = make_vae(bb, paths, fake=tiny, device=device, dtype=dtype)
    text = TextEmbeddingProvider(bb, paths if not tiny else None, fake=tiny,
                                 device=device, dtype=dtype)

    from phantom.model.hht.hht import HHT
    from phantom.model.rf import PhantomRectifiedFlow
    hht = HHT(hw, bb, mc, vae, student=student).to(device=device, dtype=dtype)
    rf = PhantomRectifiedFlow(net, hht, hw, bb, mc, layout, vae, text) \
        .to(device=device, dtype=dtype)
    n_train = sum(p.numel() for p in rf.parameters() if p.requires_grad)
    log.info("PhantomRectifiedFlow built: %.1f M trainable params", n_train / 1e6)
    return PhantomModel(rf=rf, layout=layout, bb=bb, mc=mc, hw=hw)
