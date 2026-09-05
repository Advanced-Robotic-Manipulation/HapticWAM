"""Issue #10: evaluation clocks, canonical onsets, binary intervals."""
from __future__ import annotations

import json
import numpy as np
import pytest

from phantom.config.hardware import load_hardware
from phantom.config.model import EVENT_IDX
from phantom.eval.aggregate import cell_stats, wilson_interval
from phantom.eval.metrics import acc_lead_times, merge_onsets, tactile_stream


class _Ep:
    """EpisodeReader stand-in: per-stream (ts, events)."""
    def __init__(self, streams: dict):
        self.s = streams

    def has(self, name):
        return name in self.s

    def _g(self, name):
        return {"data": np.asarray(self.s[name][1])}

    def ts(self, name):
        return np.asarray(self.s[name][0], dtype=float)


def _episode(tmp_path, offset, crossings_host, onsets_by_pad, hw):
    trace = []
    t = 0.0
    for c in crossings_host:
        trace.append({"t": c - 0.1, "gate": 0.2})
        trace.append({"t": c, "gate": 0.9})
    (tmp_path / "planner_trace.json").write_text(json.dumps(trace))
    (tmp_path / "meta.json").write_text(json.dumps({"clock_calibration": {"offset": offset}}))
    streams = {}
    for pad, onsets in zip(hw.tactile.sensors, onsets_by_pad):
        ts = np.arange(0, 20, 0.125)
        ev = np.full(len(ts), EVENT_IDX["none"])
        for t_on in onsets:
            ev[int(round(t_on / 0.125))] = EVENT_IDX["onset"]
        streams[tactile_stream(pad.name, "events")] = (ts, ev)
    return _Ep(streams)


def test_lead_time_invariant_to_clock_origin(tmp_path):
    """Shifting the host clock origin (with the matching calibration) must
    leave the physical lead time unchanged."""
    hw = load_hardware("configs/hardware.nuc.mock.yaml")
    a = tmp_path / "a"; b = tmp_path / "b"; a.mkdir(); b.mkdir()
    # master onset at 5.0 s; gate crossed at master 3.5 s
    ep_a = _episode(a, offset=0.0, crossings_host=[3.5], onsets_by_pad=[[5.0], []], hw=hw)
    ep_b = _episode(b, offset=1000.0, crossings_host=[3.5 - 1000.0], onsets_by_pad=[[5.0], []], hw=hw)
    la = acc_lead_times(a, ep_a, hw)
    lb = acc_lead_times(b, ep_b, hw)
    assert la == pytest.approx([1.5]) and lb == pytest.approx([1.5])


def test_bilateral_onset_counts_once(tmp_path):
    hw = load_hardware("configs/hardware.nuc.mock.yaml")
    ep = _episode(tmp_path, 0.0, crossings_host=[3.5],
                  onsets_by_pad=[[5.0], [5.125]], hw=hw)      # both pads, 125 ms apart
    leads = acc_lead_times(tmp_path, ep, hw)
    assert len(leads) == 1 and leads[0] == pytest.approx(1.5)
    # two DISTINCT contacts stay two
    ep2 = _episode(tmp_path, 0.0, crossings_host=[3.5, 8.0],
                   onsets_by_pad=[[5.0], [9.0]], hw=hw)
    assert len(acc_lead_times(tmp_path, ep2, hw)) == 2


def test_no_onset_or_no_crossing_is_empty_not_perfect(tmp_path):
    hw = load_hardware("configs/hardware.nuc.mock.yaml")
    ep = _episode(tmp_path, 0.0, crossings_host=[3.5], onsets_by_pad=[[], []], hw=hw)
    assert acc_lead_times(tmp_path, ep, hw) == []
    ep = _episode(tmp_path, 0.0, crossings_host=[], onsets_by_pad=[[5.0], []], hw=hw)
    assert acc_lead_times(tmp_path, ep, hw) == []


def test_merge_onsets():
    assert merge_onsets([]).size == 0
    assert merge_onsets([5.0, 5.1, 5.2, 9.0]).tolist() == [5.0, 9.0]


def test_wilson_is_nonempty_at_zero_and_all_successes():
    lo, hi = wilson_interval(0, 16)
    assert lo == 0.0 and 0.15 < hi < 0.25
    lo, hi = wilson_interval(16, 16)
    assert 0.75 < lo < 0.85 and hi == 1.0
    assert all(np.isnan(wilson_interval(0, 0)))


def test_cell_stats_reports_wilson_and_cluster_bootstrap_separately():
    rows = [{"success": "False", "damage": "False", "seed": str(i % 4)} for i in range(16)]
    st = cell_stats(rows)
    assert st["success"] == 0.0 and st["success_ci_lo"] == 0.0 and st["success_ci_hi"] > 0.1
    assert st["n_seeds"] == 4 and st["ci_method"] == "wilson"
    assert st["success_boot_ci_lo"] == 0.0 and st["success_boot_ci_hi"] == 0.0   # the bootstrap's known limit, reported as such
    one_seed = [{"success": "True", "damage": "False", "seed": "0"} for _ in range(5)]
    st1 = cell_stats(one_seed)
    assert np.isnan(st1["success_boot_ci_lo"]) and st1["n_seeds"] == 1
    assert st1["success_ci_lo"] > 0.5


def test_event_f1_uses_the_master_clock(tmp_path):
    """Same clock bug as lead times: a host-time trace against master-time
    events. With a 1000 s offset the old code clipped every row to index 0."""
    from phantom.eval.metrics import event_f1
    hw = load_hardware("configs/hardware.nuc.mock.yaml")
    n_evt = len(EVENT_IDX)
    p_onset = [0.0] * n_evt; p_onset[EVENT_IDX["onset"]] = 1.0
    p_none = [0.0] * n_evt; p_none[EVENT_IDX["none"]] = 1.0
    # onset at master 5.0; the planner predicts onset at host 5.0-1000, none elsewhere
    trace = [{"t": 2.0 - 1000.0, "p_evt": p_none}, {"t": 5.0 - 1000.0, "p_evt": p_onset},
             {"t": 8.0 - 1000.0, "p_evt": p_none}]
    (tmp_path / "planner_trace.json").write_text(json.dumps(trace))
    (tmp_path / "meta.json").write_text(json.dumps({"clock_calibration": {"offset": 1000.0}}))
    ts = np.arange(0, 20, 0.125)
    ev = np.full(len(ts), EVENT_IDX["none"]); ev[int(5.0 / 0.125)] = EVENT_IDX["onset"]
    ep = _Ep({tactile_stream(hw.tactile.sensors[0].name, "events"): (ts, ev)})
    assert event_f1(ep, hw, tmp_path) == pytest.approx(1.0)
