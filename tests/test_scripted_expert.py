"""Scripted sim expert: phases, speed limits, plan frame (CPU only, numpy)."""
from types import SimpleNamespace
import numpy as np

from phantom.sim.scripted_expert import ExpertParams, ScriptedExpertPolicy, PHASES


def drive(policy, tcp0, packet, seconds=60.0, dt=0.25):
    """Idealised executor: apply the first dt worth of each plan, then replan."""
    tcp = np.array(tcp0, dtype=float); t = 0.0; trace = []
    while t < seconds and policy.phase != "done":
        plan = policy.replan(SimpleNamespace(t=t), None, tcp)
        assert plan.actions.shape == (16, 7) and np.isfinite(plan.actions).all()
        steps = plan.actions[:int(round(dt * 16)), :6]
        assert np.all(np.linalg.norm(steps[:, :3], axis=1) <= 0.12 / 16 + 1e-9)   # per-step speed cap
        tcp[:6] += steps.sum(axis=0)
        trace.append((t, policy.phase, tcp[:3].copy(), float(plan.actions[0, 6]), plan.p_evt.argmax()))
        t += dt
    return trace


def test_expert_runs_through_all_phases_and_places_in_the_bin():
    packet = np.array([-0.399, -0.273, 0.0355])
    bin_xy = np.array([-0.3905, 0.0649])
    pol = ScriptedExpertPolicy(lambda: packet, bin_xy, ExpertParams())
    trace = drive(pol, [-0.386, -0.315, 0.351, -1.02, -1.86, 1.54], packet)
    phases = [ph for _, ph, _, _, _ in trace]
    assert pol.phase == "done"
    assert all(p in PHASES for p in phases) and phases.index("close") < phases.index("lift") < phases.index("open")
    grasp = next(pos for _, ph, pos, _, _ in trace if ph == "close")
    assert abs(grasp[2] - (packet[2] + .035)) < .01 and np.linalg.norm(grasp[:2] - packet[:2]) < .01
    carry_z = max(pos[2] for _, ph, pos, _, _ in trace if ph == "carry")
    assert .32 <= carry_z <= .34
    release = next(pos for _, ph, pos, _, _ in trace if ph == "open")
    assert abs(release[2] - .11) < .01 and abs(release[0] - bin_xy[0]) < .01 and abs(release[1] - (bin_xy[1] - .015)) < .01
    grips = {ph: g for _, ph, _, g, _ in trace}
    assert abs(grips["descend"] - .15) < 1e-6 and abs(grips["lift"] - .62) < 1e-6 and abs(grips["retreat"] - .15) < 1e-6
    events = {ph: e for _, ph, _, _, e in trace}
    assert events["carry"] == 2 and events["descend"] == 0 and events["open"] == 4   # hold / none / release


def test_slow_band_near_contact():
    packet = np.array([-0.40, -0.27, 0.0355]); pol = ScriptedExpertPolicy(lambda: packet, (-0.39, 0.06))
    pol.phase, pol.phase_t0 = "descend", 0.0
    high = pol.replan(SimpleNamespace(t=0.0), None, np.array([-0.40, -0.27, 0.15, 0, 0, 0]))
    low = pol.replan(SimpleNamespace(t=0.1), None, np.array([-0.40, -0.27, 0.09, 0, 0, 0]))
    assert abs(high.actions[0, 2]) > abs(low.actions[0, 2]) and abs(low.actions[0, 2]) <= 0.03 / 16 + 1e-9
