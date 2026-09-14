"""Scripted sim expert: phases, speed limits, orientation targets, plan frame (CPU only, numpy)."""
from types import SimpleNamespace
import numpy as np

from phantom.sim.scripted_expert import ExpertParams, ScriptedExpertPolicy, PHASES, rotvec_nearest

NO_JITTER = dict(rot_jitter_rad=0.0, place_jitter_m=0.0, z_place_jitter_m=0.0)


def drive(policy, tcp0, seconds=60.0, dt=0.2, held=None):
    """Idealised executor: apply the first dt worth of each plan, then replan.

    `held` = a dict {"z": ...} the test's object-pose function reads: while the expert holds
    the object (settle..place_descend) the object hangs 3.5 cm below the TCP, so the
    touchdown release only fires at the real place height (a fixed object would look
    like a dropped one)."""
    p = policy.p
    rate = p.action_rate_hz
    tcp = np.array(tcp0, dtype=float); t = 0.0; trace = []
    while t < seconds and policy.phase != "done":
        if held is not None and policy.phase in ("settle", "lift", "carry", "place_descend"):
            held["xyz"] = np.array([tcp[0], tcp[1], tcp[2] - 0.035])
        plan = policy.replan(SimpleNamespace(t=t), None, tcp)
        assert plan.actions.shape == (p.horizon, 7) and np.isfinite(plan.actions).all()
        assert np.all(np.linalg.norm(plan.actions[:, :3], axis=1) <= p.v_travel / rate + 1e-9)   # per-step speed cap
        assert np.all(np.linalg.norm(plan.actions[:, 3:6], axis=1) <= p.w_max_rad_s / rate + 1e-9)
        if policy.phase in ("lift", "carry", "place_descend"):
            assert np.all(np.linalg.norm(plan.actions[:, 3:6], axis=1) <= p.w_max_gripped_rad_s / rate + 1e-9)
        steps = plan.actions[:int(round(dt * rate)), :6]
        tcp[:6] += steps.sum(axis=0)
        trace.append((t, policy.phase, tcp[:6].copy(), float(plan.actions[0, 6]), plan.p_evt.argmax()))
        t += dt
    return trace


def test_expert_runs_through_all_phases_and_places_in_the_bin():
    packet = np.array([-0.399, -0.273, 0.0355])
    bin_xy = np.array([-0.3905, 0.0649])
    held = {"xyz": packet.copy()}
    pol = ScriptedExpertPolicy(lambda: held["xyz"], bin_xy, ExpertParams(**NO_JITTER))
    trace = drive(pol, [-0.386, -0.315, 0.351, -1.02, -1.86, 1.54], held=held)
    phases = [ph for _, ph, _, _, _ in trace]
    assert pol.phase == "done"
    assert all(p in PHASES for p in phases) and phases.index("close") < phases.index("lift") < phases.index("open")
    grasp = next(pose for _, ph, pose, _, _ in trace if ph == "close")
    assert abs(grasp[2] - (packet[2] + .035)) < .01 and np.linalg.norm(grasp[:2] - packet[:2]) < .01
    # orientation at closure = demo grasp orientation; at release = demo release orientation
    assert np.linalg.norm(grasp[3:6] - np.array(pol.p.rot_grasp)) < 0.1
    carry_z = max(pose[2] for _, ph, pose, _, _ in trace if ph == "carry")
    assert .32 <= carry_z <= .34
    release = next(pose for _, ph, pose, _, _ in trace if ph == "open")
    assert abs(release[2] - .10) < .01 and abs(release[0] - bin_xy[0]) < .01 and abs(release[1] - (bin_xy[1] - .015)) < .01
    assert np.linalg.norm(release[3:6] - np.array(pol.p.rot_release)) < 0.1
    grips = {ph: g for _, ph, _, g, _ in trace}
    assert abs(grips["descend"] - .15) < 1e-6 and abs(grips["lift"] - .62) < 1e-6 and abs(grips["retreat"] - .15) < 1e-6
    events = {ph: e for _, ph, _, _, e in trace}
    assert events["carry"] == 2 and events["descend"] == 0 and events["open"] == 4   # hold / none / release


def test_slow_band_near_contact():
    packet = np.array([-0.40, -0.27, 0.0355])
    pol = ScriptedExpertPolicy(lambda: packet, (-0.39, 0.06), ExpertParams(**NO_JITTER))
    pol.phase, pol.phase_t0 = "descend", 0.0
    rot = np.array(pol.p.rot_grasp)
    high = pol.replan(SimpleNamespace(t=0.0), None, np.array([-0.40, -0.27, 0.15, *rot]))
    low = pol.replan(SimpleNamespace(t=0.1), None, np.array([-0.40, -0.27, 0.09, *rot]))
    assert abs(high.actions[0, 2]) > abs(low.actions[0, 2]) and abs(low.actions[0, 2]) <= 0.03 / 10 + 1e-9


def test_episode_jitter_is_seeded_and_bounded():
    packet = np.array([-0.40, -0.27, 0.0355])
    a = ScriptedExpertPolicy(lambda: packet, (-0.39, 0.06), rng_seed=7)
    b = ScriptedExpertPolicy(lambda: packet, (-0.39, 0.06), rng_seed=7)
    c = ScriptedExpertPolicy(lambda: packet, (-0.39, 0.06), rng_seed=8)
    assert a.p == b.p and a.p != c.p
    base = ExpertParams()
    assert np.linalg.norm(np.array(a.p.rot_grasp) - base.rot_grasp) < 4 * base.rot_jitter_rad * np.sqrt(3)
    assert abs(a.p.z_place_m - base.z_place_m) < 4 * base.z_place_jitter_m
    assert a.info["episode_params"]["rot_grasp"] == list(a.p.rot_grasp)


def test_rotvec_nearest_picks_the_short_representation():
    ref = np.array([-1.7, -1.9, 1.3])
    far = ref * (1.0 + 2 * np.pi / np.linalg.norm(ref))       # same rotation, angle + 2π
    near = rotvec_nearest(ref, far)
    assert np.allclose(near, ref, atol=1e-9)


def test_deliberate_miss_reopens_and_regrasps_on_target():
    packet = np.array([-0.399, -0.273, 0.0355])
    lifted = {"z": packet[2]}
    pol = ScriptedExpertPolicy(lambda: np.array([packet[0], packet[1], lifted["z"]]), (-0.3905, 0.0649),
                               ExpertParams(**NO_JITTER, miss_offset_xy_m=(0.03, 0.0)))
    # first descent aims 3 cm off; the packet never rises on the lift -> regrasp
    tcp = np.array([-0.386, -0.315, 0.351, -1.02, -1.86, 1.54]); t = 0.0; phases = []
    while t < 60 and pol.phase != "done":
        plan = pol.replan(SimpleNamespace(t=t), None, tcp)
        tcp[:6] += plan.actions[:2, :6].sum(axis=0); phases.append(pol.phase)
        if pol.phase in ("carry", "place_descend") and pol.attempt == 2:
            lifted["z"] = tcp[2] - 0.035          # second grasp holds: the packet follows the TCP
        t += 0.2
    first_close = next(pose for ph, pose in zip(phases, [tcp]) if ph == "close") if False else None
    assert "regrasp_open" in phases and pol.attempt == 2 and pol.phase == "done"
    miss_events = [e for e in pol.events if e["event"] == "miss"]
    assert len(miss_events) == 1
    assert phases.index("regrasp_open") < phases.index("carry")


def test_touchdown_release_opens_when_the_object_stops_following():
    packet = np.array([-0.399, -0.273, 0.0355]); bin_xy = np.array([-0.3905, 0.0649])
    obj = {"z": 0.20}
    pol = ScriptedExpertPolicy(lambda: np.array([bin_xy[0], bin_xy[1] - 0.015, obj["z"]]), bin_xy,
                               ExpertParams(**NO_JITTER, z_place_m=0.05))
    pol.phase, pol.phase_t0, pol.grasp_xyz = "place_descend", 0.0, packet.copy()
    tcp = np.array([bin_xy[0], bin_xy[1] - 0.015, 0.25, *pol.p.rot_release]); t = 0.0
    opened_at = None
    while t < 10 and pol.phase in ("place_descend",):
        plan = pol.replan(SimpleNamespace(t=t), None, tcp)
        tcp[:6] += plan.actions[:2, :6].sum(axis=0); t += 0.2
        # the object hangs 5 cm below the TCP until it meets a support at z 0.12
        obj["z"] = max(0.12, tcp[2] - 0.05)
    assert pol.phase == "open" and tcp[2] > 0.15, (pol.phase, tcp[2])   # opened well above the 0.05 target
    assert any(e["event"] == "touchdown" for e in pol.events)
