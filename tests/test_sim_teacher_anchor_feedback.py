"""Delivery-only veto overlay preserves historical rules and v1 release masking."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from test_sim_teacher_anchor_finish import module
from test_sim_terminal_veto import Feedback, action, hardware, proposal, snapshot

from tools.sim.prepare_teacher_anchor_feedback import BASE_SHA, MODULE, prepare, sha

FIXTURE = Path(__file__).parent / "fixtures/teacher_anchor_v1/deployment_filters.py"


@pytest.fixture
def implementations(tmp_path, monkeypatch):
    base, output = tmp_path / "base", tmp_path / "overlay"
    source = base / MODULE
    source.parent.mkdir(parents=True)
    source.write_text(FIXTURE.read_text())
    manifest = prepare(base, output)
    old = module(monkeypatch, "anchor_filter_original", source).TerminalVetoFilter
    new = module(
        monkeypatch, "anchor_filter_delivery", output / MODULE
    ).TerminalVetoFilter
    return old, new, base, output, manifest


class CurrentRing:
    def __init__(self, field, value, t=0.2):
        self.field, self.value, self.t = field, np.asarray(value), t

    def latest(self, n):
        return np.array([self.t]), {self.field: self.value[None]}


def feedback(z=0.2, grip=0.29, *, release=True):
    value = Feedback()
    value._last_observe_t = 0.2
    value.rings.update(
        arm=CurrentRing("tcp_pose", [0, 0, z, 0, 0, 0]),
        gripper=CurrentRing("state", [grip, 3]),
    )
    if release:
        value.release_controller = object()
    return value


def test_only_one_overlay_module_and_no_native_release_refactor(
    implementations, tmp_path
):
    _, _, base, output, manifest = implementations
    assert sha((base / MODULE).read_text()) == BASE_SHA
    assert set(manifest["output_sha256"]) == {MODULE}
    transformed = (output / MODULE).read_text()
    assert "from phantom.deploy.release_controller" not in transformed
    assert "placement_policy_release_v1" in transformed
    with pytest.raises(FileExistsError):
        prepare(base, output)
    with pytest.raises(ValueError, match="protected"):
        prepare(base, base / "bad")
    (base / MODULE).write_text((base / MODULE).read_text() + "\n")
    with pytest.raises(ValueError, match="hashes"):
        prepare(base, tmp_path / "bad")


@pytest.mark.parametrize(("request_z", "delivery_z"), [(0.2, 0.08), (0.08, 0.2)])
def test_delivery_feedback_changes_only_veto_measurement_and_preserves_original_input(
    implementations, request_z, delivery_z
):
    old, new, *_ = implementations
    config = {"z_ref": 0.0415, "z_margin": 0.0615}
    captured = snapshot(t=0, z=request_z, grip=0.2)
    source = proposal()
    before = deepcopy(source)
    original_state = captured.ur_state.copy()
    a = old(hardware(), config, implementation="fd4a032")(
        source, captured, feedback(delivery_z)
    )
    b = new(hardware(), config, implementation="fd4a032")(
        source, captured, feedback(delivery_z)
    )
    assert action(a) == ("close_masked" if request_z > 0.103 else "close_allowed")
    assert action(b) == ("close_masked" if delivery_z > 0.103 else "close_allowed")
    rec = b.diag["terminal_veto"]
    assert rec["feedback_source"] == "current_delivery"
    assert rec["feedback_tcp_pose"][2] == delivery_z
    assert rec["feedback_gripper"] == 0.29
    assert rec["snapshot_t_s"] == 0 and rec["applied_at_s"] == 0.2
    np.testing.assert_array_equal(captured.ur_state, original_state)
    np.testing.assert_array_equal(source.actions, before.actions)
    assert source.diag == before.diag and source._cpk_token == before._cpk_token
    np.testing.assert_array_equal(a.actions[:, :6], b.actions[:, :6])
    # Without the opt-in placement controller both variants still use request.
    no_release = new(hardware(), config, implementation="fd4a032")(
        source, captured, feedback(delivery_z, release=False)
    )
    np.testing.assert_array_equal(no_release.actions, a.actions)
    assert (
        no_release.diag["terminal_veto"]["feedback_source"]
        == "request_snapshot_historical"
    )


def test_local_v1_policy_opening_passthrough_retained(implementations):
    old, new, *_ = implementations
    raw = proposal()
    raw.actions[:2, 6] = 0.3
    before = raw.actions.copy()
    results = []
    for cls in (old, new):
        current = feedback(z=0.2, grip=0.29)
        current.placement_release_opening_mask = lambda _g: np.array(
            [True, True, False, False]
        )
        results.append(
            cls(hardware(), implementation="fd4a032")(raw, snapshot(grip=0.29), current)
        )
    np.testing.assert_array_equal(results[0].actions, results[1].actions)
    np.testing.assert_array_equal(results[1].actions[:, 6], [0.3, 0.3, 0.29, 0.29])
    assert (
        results[1].diag["terminal_veto"]["placement_release_variant"]
        == "placement_policy_release_v1"
    )
    assert results[1]._cpk_token is None
    np.testing.assert_array_equal(raw.actions, before)


def test_historical_retry_stop_still_preempts_release_passthrough(implementations):
    old, new, *_ = implementations
    for cls in (old, new):
        current = feedback(z=0.2, grip=0.2)
        current._last_observe_t = 0
        current.rings["arm"].t = current.rings["gripper"].t = 0
        current.placement_release_opening_mask = lambda g: np.ones(len(g), dtype=bool)
        veto = cls(hardware(), {"max_retries": 0}, implementation="fd4a032")
        veto(proposal(p_none=0.1), snapshot(), current)
        current.history = [(0.1, 0.6)]
        current._last_observe_t = 0.2
        current.rings["gripper"].value[0] = 0.6
        current.rings["arm"].t = current.rings["gripper"].t = 0.2
        result = veto(proposal(), snapshot(t=0.2, grip=0.6), current)
        assert action(result) == "retry_cap"
        assert current.stopped_reason == "veto_retry_cap"
        assert (
            "placement_release_passthrough_indices" not in result.diag["terminal_veto"]
        )
