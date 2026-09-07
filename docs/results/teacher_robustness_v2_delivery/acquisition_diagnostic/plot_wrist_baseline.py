#!/usr/bin/env python3
"""Plot the unchanged native baseline and actual contact history."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def main():
    data = json.loads((HERE / "wrist_baseline_trace.json").read_text())
    s = data["series"]
    t = np.array(s["t_s"])
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 8), layout="constrained")
    stop = data["trigger"]["t_s"]
    onset = data["trigger"]["over_since_s"]
    for column, limits in enumerate([(0, stop + 0.1), (15, stop + 0.1)]):
        a, b, c = axes[:, column]
        a.plot(t, s["wrist_fz_n"], label="Actual proxy wrist Fz", color="#0072B2")
        a.plot(
            t,
            s["rolling_baseline_fz_n"],
            label="Native rolling baseline Fz",
            color="#D55E00",
        )
        a.axhline(
            data["recorded_initial_bias_n_nm"][2],
            ls=":",
            color="gray",
            label="Measured initial Fz bias",
        )
        a.set_ylabel("Force in world Z (N)")
        a.set_title("Full execution" if column == 0 else "Contact unloading and stop")
        b.plot(t, s["mat_normal_sum_n"], label="Finger / mat", color="#CC79A7")
        b.plot(
            t, s["packet_normal_sum_n"], label="Fingers / packet (sum)", color="#009E73"
        )
        b.plot(t, s["bench_normal_sum_n"], label="Finger / bench", color="#999999")
        b.set_ylabel("Positive contact normal sum (N)")
        c.plot(
            t,
            s["force_deviation_n"],
            color="#0072B2",
            label="Native force deviation norm",
        )
        c.axhline(
            data["force_limit_n"], ls="--", color="black", label="Unchanged 60 N limit"
        )
        c.set_ylabel("Deviation from baseline (N)")
        for ax in [a, b, c]:
            ax.axvspan(onset, stop, alpha=0.1, color="red")
            ax.axvline(
                stop, color="red", lw=1, label="Stop17.972s" if ax is c else None
            )
            ax.set_xlim(*limits)
            ax.set_xlabel("Simulation time (s)")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(
        "Native baseline absorbs gradual contact; unloading triggers the deviation guard\nv5_6 NFE1/K4 · start1787396273 · seed903101",
        fontsize=13,
    )
    fig.supxlabel(
        "Actual frozen trial; unchanged EMA2s, 60N /15N m limits, 0.3s debounce. Contact proxy is not a calibrated UR3 wrist sensor.",
        fontsize=9,
    )
    fig.savefig(HERE / "wrist_baseline_drift.png", dpi=160)
    fig.savefig(HERE / "wrist_baseline_drift.pdf")


if __name__ == "__main__":
    main()
