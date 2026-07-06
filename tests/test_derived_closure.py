"""The core data-layer test: derived.py must recover the ContactScenario's
scripted event timeline from mock-rendered tactile frames (mock <-> derived
closure), and from full synthetic episodes."""

import numpy as np

from phantom.config.model import EVENT_IDX
from phantom.data import derived as dv
from phantom.drivers.mock.dmtac import render_fields
from phantom.drivers.mock.scenario import ContactScenario


def _stack(f: dict) -> np.ndarray:
    return np.concatenate([f["deformation"], f["depth"][..., None],
                           f["shear"], f["dist_force"]], axis=-1)


def test_mock_derived_closure(small_hw):
    hw = small_hw
    scenario = ContactScenario(seed=0)
    rng = np.random.default_rng(0)
    dt = 1.0 / hw.tactile.rate_hz
    T = int(scenario.timings.cycle_s / dt)

    mask_frac = np.zeros(T, dtype=np.float32)
    slip = np.zeros(T, dtype=np.float32)
    cop_err = []
    prev = None
    for k in range(T):
        t = k * dt
        st = scenario.state(t)
        stack = _stack(render_fields(st, hw.tactile, rng))
        d = dv.derive_timestep(stack, prev, dt, hw)
        mask_frac[k] = d["mask_frac"]
        slip[k] = d["slip"]
        if st.in_contact and st.press_depth > 0.3 and not np.isnan(d["cop"]).any():
            cop_err.append(np.linalg.norm(d["cop"] - np.asarray(st.blob_center_uv)))
        prev = stack

    events = dv.event_labels(mask_frac, slip, hw.derived)
    found = {name for name, idx in EVENT_IDX.items() if (events == idx).any()}
    for must in ("none", "onset", "hold", "release"):
        assert must in found, f"missing event {must}; got {found}"

    # contact frames coincide with scripted contact
    scripted = np.array([scenario.state(k * dt).in_contact for k in range(T)])
    agree = ((mask_frac > 0) == scripted).mean()
    assert agree > 0.9, f"contact mask agrees with script only {agree:.0%}"

    # CoP tracks the blob center while firmly pressed
    assert cop_err and np.mean(cop_err) < 0.3, f"CoP error {np.mean(cop_err):.2f}"

    # slip fires in the slip phase, not (much) in hold
    slip_phase = np.array([scenario.phase_at(k * dt)[0] == "slip" for k in range(T)])
    hold_phase = np.array([scenario.phase_at(k * dt)[0] == "hold" for k in range(T)])
    assert slip[slip_phase].mean() > slip[hold_phase].mean() * 1.5


def test_gate_label_lookahead():
    mf = np.array([0, 0, 0, 0.5, 0.5, 0], dtype=np.float32)
    assert dv.contact_within(mf, 0, 3)
    assert not dv.contact_within(mf, 0, 2)
    assert not dv.contact_within(mf, 4, 1)


def test_tau_obj_calibration():
    peaks = np.array([1.0, 1.2, 0.8, 1.1, 0.9, 1.3])
    tau = dv.calibrate_tau_obj(peaks, 0.9)
    assert 1.2 <= tau <= 1.3
    pen = dv.force_safety_penalty(np.array([tau - 0.1, tau + 0.5]), tau, 2.0)
    assert pen[0] == 0.0 and np.isclose(pen[1], -1.0)


def test_synthetic_episode_roundtrip(small_hw, tmp_path):
    """Generator writes schema-conforming episodes; postprocess recovers events."""
    from phantom.data.episode_store import EpisodeReader
    from phantom.data.schema import STREAM_ACTIONS, tactile_stream
    from phantom.data.synthetic import SyntheticEpisodeGenerator
    from phantom.recording.postprocess import postprocess_episode

    gen = SyntheticEpisodeGenerator(small_hw, seed=1, rate_scale=1.0)
    ep = gen.generate(tmp_path, task="approach_contact", duration_s=7.0)
    postprocess_episode(ep, small_hw)
    r = EpisodeReader(ep)
    s0 = small_hw.tactile.sensors[0].name
    assert r.n(STREAM_ACTIONS) > 0
    assert r.n(tactile_stream(s0, "fields_ds")) > 0
    fds = r._g(tactile_stream(s0, "fields_ds"))["data"]
    assert fds.shape[1:] == (small_hw.recording.field_ds.h,
                             small_hw.recording.field_ds.w, 8)
    ev = np.asarray(r._g(tactile_stream(s0, "events"))["data"][:])
    assert (ev == EVENT_IDX["onset"]).any() and (ev == EVENT_IDX["hold"]).any()
