#!/usr/bin/env python3
"""Scientific plot from the saved audit; separate scene/gel/executor clocks."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def main():
    data = json.loads((HERE / "audit.json").read_text())
    fig, axes = plt.subplots(4, 3, figsize=(13.2, 10.8), layout="constrained")
    labels = [
        "ftA3000 K4 · start5963 · seed903101",
        "v5_6 K4 · start6028 · seed903101",
        "v5_6 K4 · start6273 · seed903101",
        "ftA1500 NFE5 · start6273 · seed903102",
    ]
    colors = ["#0072B2", "#D55E00"]
    for row, case, label in zip(axes, data["trials"], labels):
        p = case["plot"]
        acq = case["metrics"]["event_times_s"]["acquisition"]
        t, tg, te = (np.array(p[k]) for k in ["scene_t_s", "gel_t_s", "execution_t_s"])
        force = np.asarray(p["physical_pad_packet_normal_n"])
        gel = np.asarray(p["gel_force_n"])
        for side, color, name in zip(range(2), colors, ["L", "R"]):
            row[0].plot(t, force[:, side], color=color, lw=1.1, label=f"Packet {name}")
            row[0].step(
                tg,
                gel[:, side],
                where="post",
                color=color,
                ls="--",
                lw=1.5,
                label=f"Active gel {name}",
            )
        row[0].set_yscale("symlog", linthresh=0.1)
        row[0].axhline(2.5, color="gray", lw=0.7, ls=":")
        row[0].set_ylabel("Normal load (N)")
        row[0].set_title(label, fontsize=10, loc="left")
        row[1].plot(
            te, p["accepted_tcp_z_m"], color="#D55E00", ls="--", label="Target TCP z"
        )
        row[1].plot(te, p["measured_tcp_z_m"], color="#0072B2", label="Measured TCP z")
        row[1].plot(t, p["object_z_m"], color="#009E73", lw=2, label="Packet center z")
        row[1].axhline(
            case["metrics"]["object"]["initial_center_m"][2] + 0.03,
            color="#009E73",
            ls=":",
            lw=0.8,
            label="Packet +30 mm",
        )
        row[1].set_ylabel("World z (m)")
        row[2].plot(
            te,
            p["commanded_closure"],
            color="#D55E00",
            ls="--",
            label="Commanded closure",
        )
        row[2].plot(
            te, p["measured_closure"], color="#0072B2", label="Measured closure"
        )
        row[2].plot(
            te,
            [np.nan if v is None else v for v in p["latch"]],
            color="black",
            lw=2,
            ls=":",
            label="Latch",
        )
        row[2].set_ylabel("Closure (0=open, 1=closed)")
        lo, hi = max(0, acq - 1), min(acq + 4, case["execution_end_s"])
        for ax in row:
            ax.set_xlim(lo, hi)
            ax.axvline(acq, color="black", lw=0.7, ls="--")
            if case["first_safety_stop"] and case["execution_end_s"] <= hi:
                ax.axvline(case["execution_end_s"], color="red", lw=1.5)
            ax.grid(alpha=0.22)
            ax.tick_params(labelsize=8)
            ax.set_xlabel("Simulation time (s)", fontsize=9)
        # Only visible-window samples set linear plot ranges.
        for ax, groups in [
            (
                row[1],
                [
                    (te, p["accepted_tcp_z_m"]),
                    (te, p["measured_tcp_z_m"]),
                    (t, p["object_z_m"]),
                ],
            ),
            (row[2], [(te, p["commanded_closure"]), (te, p["measured_closure"])]),
        ]:
            values = np.concatenate(
                [np.asarray(v)[(times >= lo) & (times <= hi)] for times, v in groups]
            )
            span = max(float(np.ptp(values)), 0.01)
            ax.set_ylim(
                float(values.min()) - 0.1 * span, float(values.max()) + 0.15 * span
            )
        row[0].legend(fontsize=7, ncol=2, loc="upper left")
        row[1].legend(fontsize=7, loc="upper left")
        row[2].legend(fontsize=7, loc="lower left")
    fig.suptitle(
        "Bilateral acquisition did not become a sustained lift\nFixed corrected screen: four acquisition cases; no model ranking",
        fontsize=14,
    )
    fig.supxlabel(
        "Dashed vertical: acquisition onset. Red: first safety stop. Gel is an uncalibrated proxy; clocks and sampling rates remain distinct.",
        fontsize=9,
    )
    fig.savefig(HERE / "acquisition_traces.png", dpi=160)
    fig.savefig(HERE / "acquisition_traces.pdf")


if __name__ == "__main__":
    main()
