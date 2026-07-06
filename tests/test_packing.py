"""ACE fixed-packing round-trip tests (cosmos-free; torch only)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from phantom.config.backbone import BackboneConfig
from phantom.config.model import N_EVENTS, PhantomModelConfig
from phantom.model.ace.packing import ActionPacker, ContactPackage, ContactPacker
from phantom.model.sequence import SequenceLayout


@pytest.fixture
def setup(small_hw):
    bb = BackboneConfig.tiny()
    mc = PhantomModelConfig()
    layout = SequenceLayout.build(bb, mc, small_hw, student=False)
    return small_hw, bb, layout


def rand_cpk(hw, layout, B=2, gen=None) -> ContactPackage:
    g = gen or torch.Generator().manual_seed(0)
    Tc = layout.t_video_gen
    Fn = hw.n_fingers
    cph, cpw = hw.cpk_shape
    return ContactPackage(
        event=torch.randint(0, N_EVENTS, (B, Tc), generator=g),
        d_disp=torch.randn(B, Tc, Fn, 3, cph, cpw, generator=g) * 0.3,
        d_fz=torch.randn(B, Tc, Fn, cph, cpw, generator=g) * 0.3,
        mask=(torch.rand(B, Tc, Fn, cph, cpw, generator=g) > 0.7).float(),
        cop=torch.rand(B, Tc, Fn, 2, generator=g) * 1.2 - 0.6,
        slip=torch.rand(B, Tc, Fn, generator=g),
        wrench=torch.randn(B, Tc, Fn, 6, generator=g) * 0.5,
        wrist=torch.randn(B, Tc, 6, generator=g) * 0.5,
    )


def test_contact_pack_shape(setup):
    hw, bb, layout = setup
    packer = ContactPacker(hw, layout)
    cpk = rand_cpk(hw, layout)
    x = packer.pack(cpk)
    assert x.shape == (2, layout.lat_c, layout.t_video_gen, layout.lat_h, layout.lat_w)


def test_contact_roundtrip_scalars(setup):
    hw, bb, layout = setup
    packer = ContactPacker(hw, layout)
    cpk = rand_cpk(hw, layout)
    rec = packer.unpack(packer.pack(cpk))
    # tiled scalars recover near-exactly
    assert torch.allclose(rec.wrench, cpk.wrench, atol=1e-3)
    assert torch.allclose(rec.wrist, cpk.wrist, atol=1e-3)
    assert torch.allclose(rec.slip, cpk.slip, atol=1e-3)
    # event argmax recovers exactly
    assert (rec.event_probs().argmax(-1) == cpk.event).all()
    # CoP recovers to within the bump width
    err = (rec.cop - cpk.cop).norm(dim=-1)
    assert float(err.mean()) < 0.25


def test_contact_roundtrip_fields(setup):
    hw, bb, layout = setup
    packer = ContactPacker(hw, layout)
    cpk = rand_cpk(hw, layout)
    # smooth fields survive the resample round trip much better than noise:
    cpk = ContactPackage(
        event=cpk.event, cop=cpk.cop, slip=cpk.slip, wrench=cpk.wrench,
        wrist=cpk.wrist, mask=cpk.mask,
        d_disp=torch.nn.functional.avg_pool2d(
            cpk.d_disp.flatten(0, 2), 5, 1, 2).reshape_as(cpk.d_disp),
        d_fz=torch.nn.functional.avg_pool2d(
            cpk.d_fz.flatten(0, 2).unsqueeze(1), 5, 1, 2)[:, 0].reshape_as(cpk.d_fz),
    )
    rec = packer.unpack(packer.pack(cpk))
    corr = np.corrcoef(rec.d_fz.flatten().numpy(), cpk.d_fz.flatten().numpy())[0, 1]
    assert corr > 0.9, f"d_fz field correlation {corr:.2f}"


def test_summary_dim(setup):
    hw, bb, layout = setup
    packer = ContactPacker(hw, layout)
    cpk = rand_cpk(hw, layout)
    s = packer.flatten_summary(cpk)
    assert s.shape == (2, packer.summary_dim())


def test_action_roundtrip(setup):
    hw, bb, layout = setup
    packer = ActionPacker(hw, layout)
    g = torch.Generator().manual_seed(1)
    a = torch.randn(3, hw.control.chunk_horizon, hw.control.action_dim, generator=g)
    rec = packer.unpack(packer.pack(a))
    assert rec.shape == a.shape
    assert torch.allclose(rec, a, atol=1e-4)


def test_action_unpack_denoises(setup):
    hw, bb, layout = setup
    packer = ActionPacker(hw, layout)
    g = torch.Generator().manual_seed(2)
    a = torch.randn(1, hw.control.chunk_horizon, hw.control.action_dim, generator=g)
    x = packer.pack(a)
    noisy = x + torch.randn(x.shape, generator=g) * 0.1
    rec = packer.unpack(noisy)
    assert (rec - a).abs().mean() < 0.05  # strip-mean averages the noise away
