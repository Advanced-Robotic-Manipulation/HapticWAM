"""Frozen-v1 parity until FINISH, measured hold, and preserved safety priority."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from phantom_test_utils import make_small_hw
from test_sim_placement_release import CONFIG, INSIDE, observe, plan

from tools.sim.prepare_teacher_anchor_finish import (
    BASE_HASHES,
    DONOR_HASHES,
    prepare,
    sha,
)

FIXTURES = Path(__file__).parent / "fixtures/teacher_anchor_v1"
DONOR_FIXTURES = Path(__file__).parent / "fixtures/teacher_anchor_finish_donor"


@pytest.fixture
def trees(tmp_path):
    base, overlay, donor = (tmp_path / name for name in ("base", "overlay", "donor"))
    # Both source sides are historical evidence. Current implementation changes
    # must not replace the reviewed donor or weaken the assembler's hash guard.
    for tree, hashes, fixtures in (
        (base, BASE_HASHES, FIXTURES),
        (donor, DONOR_HASHES, DONOR_FIXTURES),
    ):
        for relative in hashes:
            path = tree / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text((fixtures / (path.name + ".txt")).read_text())
    manifest = prepare(base, overlay, donor)
    return base, overlay, manifest


def module(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, result)
    spec.loader.exec_module(result)
    return result


def adapter(monkeypatch, tree, *, finish=False):
    label = "overlay" if finish else tree.name
    release = module(
        monkeypatch,
        "anchor_release_" + label,
        tree / "phantom/sim/release_controller.py",
    )
    implementation = module(
        monkeypatch, "anchor_adapter_" + label, tree / "phantom/sim/policy_adapter.py"
    )
    hw = make_small_hw(
        safety={
            "wrist_extension_stop_m": None,
            "reach_clamp_m": None,
            "workspace_m": {"x": [-0.7, 0.15], "y": [-0.5, 0.3], "z": [0.03, 0.8]},
        }
    )
    with monkeypatch.context() as scoped:
        scoped.setitem(sys.modules, "phantom.sim.release_controller", release)
        return implementation.SimulationPolicyAdapter(
            hw,
            SimpleNamespace(),
            mode="student",
            release_config={
                **CONFIG,
                **({"finish_after_release": True} if finish else {}),
            },
        )


def execute(ad, i, *, reclose=False, qd=None):
    t = i * 0.008
    load = 0 if i == 0 or 40 <= i < 200 else 3
    measured = 0.3 if 40 <= i < 200 else 0.55
    observe(ad, t, load=load, measured=measured, qd=qd)
    if i in (0, 2) or (reclose and i == 160):
        p = plan(t, 0.63 if i in (0, 160) else 0.4)
        p.actions[:, 0] = 0.0008
        assert ad.submit(p, t)
    cmd = ad.step(t)
    ad.report_execution(
        t, accepted=True, tcp_pose=cmd.tcp_pose, gripper_command=cmd.gripper
    )
    return cmd


def same_command(a, b):
    np.testing.assert_array_equal(a.tcp_pose, b.tcp_pose)
    assert (a.gripper, a.dt, a.stopped, a.reason) == (
        b.gripper,
        b.dt,
        b.stopped,
        b.reason,
    )
    assert a.diagnostics["safety_events"] == b.diagnostics["safety_events"]
    assert a.diagnostics["play_time_s"] == b.diagnostics["play_time_s"]


def test_overlay_changes_only_two_modules_and_keeps_base_immutable(trees):
    base, overlay, manifest = trees
    assert {str(p.relative_to(overlay)) for p in overlay.rglob("*") if p.is_file()} == {
        *BASE_HASHES,
        "anchor_finish_overlay.json",
    }
    for name, digest in BASE_HASHES.items():
        assert sha((base / name).read_text()) == digest
        assert sha((overlay / name).read_text()) == manifest["output_sha256"][name]
    # Local constructor and original-veto eligibility must retain v1 imports.
    source = (overlay / "phantom/sim/policy_adapter.py").read_text()
    assert "from phantom.deploy.release_controller" not in source
    assert "from phantom.sim.release_controller" in source


def test_disabled_finish_matches_v1_through_release_and_rearm(trees, monkeypatch):
    base, overlay, _ = trees
    old, new = adapter(monkeypatch, base), adapter(monkeypatch, overlay)
    for i in range(300):
        same_command(execute(old, i, reclose=True), execute(new, i, reclose=True))
        assert old.release_controller.phase == new.release_controller.phase
        assert old._grip_latch == new._grip_latch
    assert new.completed_reason is None
    assert old.release_controller.phase == "holding"


def test_enabled_finish_identical_until_unloaded_then_holds_measured_pose(
    trees, monkeypatch
):
    base, overlay, _ = trees
    old, new = adapter(monkeypatch, base), adapter(monkeypatch, overlay, finish=True)
    for i in range(200):
        a, b = execute(old, i), execute(new, i)
        if new.completed_reason:
            break
        same_command(a, b)
    assert new.completed_reason == "placement_release_finished"
    assert new.release_controller.committed_at is not None
    assert new.release_controller.finished_at > new.release_controller.committed_at
    assert new.safety.contact_load == {"left": 0.0, "right": 0.0}
    assert new.completed_at_s == b.t
    np.testing.assert_array_equal(b.tcp_pose, INSIDE)
    assert b.gripper == 0.4 and not b.stopped
    assert not new.ready_for_replan(b.t + 0.008)
    assert not new.submit(plan(b.t, 0.8), b.t)
    with pytest.raises(RuntimeError, match="episode ended"):
        new.replan(t=b.t)
    # Subsequent measured robot drift does not change the achieved hold target.
    t = b.t + 0.008
    observe(new, t, tcp=INSIDE + [0.003, 0, 0, 0, 0, 0], load=0, measured=0.3)
    hold = new.step(t)
    new.report_execution(t, accepted=True, gripper_command=hold.gripper)
    np.testing.assert_array_equal(hold.tcp_pose, INSIDE)
    # A live guard still preempts the completion hold immediately.
    observe(new, t + 0.008, load=0, measured=0.3, qd=np.ones(6) * 3)
    stopped = new.step(t + 0.008)
    assert stopped.stopped and "joint_speed" in stopped.diagnostics["safety_events"]
    assert stopped.diagnostics["completed_reason"] == "placement_release_finished"


def test_no_finish_without_loaded_latch_and_policy_release(trees, monkeypatch):
    _, overlay, _ = trees
    ad = adapter(monkeypatch, overlay, finish=True)
    for i in range(200):
        t = i * 0.008
        observe(ad, t, load=0, measured=0.3)
        if i == 0:
            ad.submit(plan(t, 0.4), t)
        cmd = ad.step(t)
        ad.report_execution(t, accepted=True, gripper_command=cmd.gripper)
    assert ad.completed_reason is None and ad.release_controller.phase == "unarmed"


def test_hash_guard_dry_run_and_output_refusal(trees, tmp_path):
    base, overlay, manifest = trees
    donor = Path(manifest["donor_source"])
    preview = tmp_path / "preview"
    assert prepare(base, preview, donor, dry_run=True)["dry_run"] and not preview.exists()
    with pytest.raises(FileExistsError):
        prepare(base, overlay, donor)
    with pytest.raises(ValueError, match="protected source"):
        prepare(base, base / "bad", donor)
    source = base / "phantom/sim/policy_adapter.py"
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="Unreviewed source hash"):
        prepare(base, preview, donor)
    assert not preview.exists()
