#!/usr/bin/env python3
"""CPU wrist-subguard replay; does not drive a robot or rerun a policy.

Run from the repository: PYTHONPATH=. python docs/results/wrist_baseline_20260907/replay_6273.py
The continued saved trajectory after each counterfactual stop is descriptive,
not a possible closed-loop outcome after the different stop time.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from phantom.config.hardware import load_hardware
from phantom.deploy.safety import SafetyMonitor, apply_wrench_baseline_mode

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def legacy_trace(t, wrist, hw):
    """Frozen 16567f3 wrist-block arithmetic, independent of amended monitor."""
    base, last, over_since = None, None, None
    rows = []
    sf = hw.safety
    for now, ft in zip(t, wrist, strict=True):
        if base is None:
            base = ft.copy()
        before = base.copy()
        dev = ft - base
        f_mag = float(np.linalg.norm(dev[:3]))
        t_mag = float(np.linalg.norm(dev[3:]))
        dt = now - last if last is not None else 0.0
        last = now
        stop = False
        if f_mag > sf.wrench_limit_N or t_mag > sf.wrench_limit_Nm:
            if over_since is None:
                over_since = now
            if now - over_since >= sf.wrench_debounce_ticks / hw.control.action_rate_hz:
                stop = True
        else:
            over_since = None
            if dt > 0.0:
                alpha = min(1.0, dt / sf.wrench_baseline_tau_s)
                base += alpha * (ft - base)
        rows.append(
            {
                "reference_before": before.tolist(),
                "reference_after": base.tolist(),
                "deviation": dev.tolist(),
                "force_deviation_n": f_mag,
                "torque_deviation_nm": t_mag,
                "over_since_s": over_since,
                "stop": stop,
            }
        )
    return rows


def audit():
    provenance = json.loads((HERE / "6273_wrist_trace_provenance.json").read_text())
    path = HERE / "6273_wrist_trace.npz"
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == provenance["extracted_npz_sha256"]
    )
    with np.load(path) as data:
        t, sample_t, wrist = (data[k] for k in ("t", "sample_t", "wrist"))
    hw = load_hardware(ROOT / "configs/hardware.nuc.yaml", quiet=True)
    assert (
        hw.safety.wrench_limit_N,
        hw.safety.wrench_limit_Nm,
        hw.safety.wrench_baseline_tau_s,
        hw.safety.wrench_debounce_ticks / hw.control.action_rate_hz,
    ) == (60, 15, 2, 0.3)
    legacy = legacy_trace(t, wrist, hw)
    output = {
        "schema": 1,
        "source": provenance,
        "scope": "saved-input wrist subguard only; no policy/dynamics counterfactual",
        "parameters": {
            "force_limit_n": 60,
            "torque_limit_nm": 15,
            "rolling_tau_s": 2,
            "debounce_s": 0.3,
        },
        "rows": len(t),
        "initial_wrist": wrist[0].tolist(),
        "modes": {},
    }
    for mode in ("rolling_calm", "episode_fixed"):
        mon = SafetyMonitor(apply_wrench_baseline_mode(hw, mode), {})
        first_crossing = first_stop = None
        max_force = max_torque = max_drift = 0.0
        for index, (now, stamp, ft) in enumerate(zip(t, sample_t, wrist, strict=True)):
            event = mon._check_wrench(float(now), float(stamp), ft)
            report = mon.wrench_diagnostics()
            assert report["valid"]
            if mode == "rolling_calm":
                for key, value in legacy[index].items():
                    actual = event is not None if key == "stop" else report[key]
                    assert actual == value, (index, key, actual, value)
            if report["over_since_s"] is not None and first_crossing is None:
                first_crossing = float(now)
            if event and first_stop is None:
                first_stop = {"t_s": float(now), "decision": report}
            max_force = max(max_force, report["force_deviation_n"])
            max_torque = max(max_torque, report["torque_deviation_nm"])
            max_drift = max(
                max_drift,
                float(
                    np.linalg.norm(
                        np.asarray(report["reference_after"][:3]) - wrist[0, :3]
                    )
                ),
            )
        output["modes"][mode] = {
            "first_instantaneous_crossing_s": first_crossing,
            "first_debounced_stop": first_stop,
            "force_deviation_max_n": max_force,
            "torque_deviation_max_nm": max_torque,
            "reference_force_drift_max_n": max_drift,
            "at_recorded_stop": report,
        }
    output["default_parity"] = {
        "bitwise_numeric_equal_all_2247_rows": True,
        "compared": list(legacy[0]),
        "frozen_reference_commit": "16567f3",
    }
    output["source_sha256"] = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            ROOT / "phantom/config/hardware.py",
            ROOT / "phantom/deploy/safety.py",
            Path(__file__),
        ]
    }
    return output


if __name__ == "__main__":
    result = audit()
    (HERE / "6273_replay_audit.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                mode: r["first_debounced_stop"]["t_s"]
                for mode, r in result["modes"].items()
            }
        )
    )
