"""Day-1 hardware bench (pipeline.md verification checklist item 1 + 2).

Semi-automated: runs the measurable procedures against the REAL drivers and
prints the exact configs/hardware.yaml fields to update. It never edits the
YAML itself — review the numbers, edit the file, set meta.bench_verified: true.

    python -m phantom.scripts.bench_day1 [--items a b c d arm]

Items (docs/hardware_bench_day1.md has the full procedures):
  a    force units      -> tactile.force_unit_to_N / dist_force_unit_to_N
  b    shapes + rates   -> tactile.raw_img/infer_img/field, tactile.rate_hz
  c    H/W ordering     -> tactile.hw_order (needs a known-corner press)
  d    throughput       -> recording.* feasibility
  arm  generation/F-T   -> arm.generation, wrist_ft.source/rate_hz
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from phantom.config.hardware import load_hardware

log = logging.getLogger("bench_day1")


def bench_b_shapes_rates(hw) -> None:
    from phantom.drivers.real.dmtac import DmTacSensor
    print("\n=== (b) shapes + concurrent rates ===")
    sensors = [DmTacSensor(hw.tactile, s) for s in hw.tactile.sensors]
    for s in sensors:
        s.connect()
    try:
        f = sensors[0].read()
        print(f"  infer_img shape: {None if f.infer_img is None else f.infer_img.shape}"
              f"  -> tactile.infer_img {{h, w, c}}")
        raw = sensors[0].read_raw_img()
        print(f"  raw_img shape:   {raw.shape}  -> tactile.raw_img")
        print(f"  deformation:     {f.deformation.shape}  depth: {f.depth.shape}"
              f"  -> tactile.field (verify order with item c!)")
        print(f"  wrench: {f.wrench.shape}  area: {f.contact_area_mm2:.1f} mm^2")
        # concurrent dual-sensor rate over 5 s
        n, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < 5.0:
            for s in sensors:
                s.read()
            n += 1
        rate = n / (time.perf_counter() - t0)
        print(f"  sustained concurrent rate: {rate:.1f} Hz/sensor "
              f"-> tactile.rate_hz (configured {hw.tactile.rate_hz})")
    finally:
        for s in sensors:
            s.disconnect()


def bench_c_ordering(hw) -> None:
    from phantom.drivers.real.dmtac import DmTacSensor
    print("\n=== (c) numpy H/W ordering ===")
    print("  Press a fingertip near the corner that is TOP-LEFT when the sensor")
    print("  is viewed in its mounted orientation, hold, then press Enter.")
    input("  ready? ")
    s = DmTacSensor(hw.tactile, hw.tactile.sensors[0])
    s.connect()
    try:
        f = s.read()
        idx = np.unravel_index(np.abs(f.depth).argmax(), f.depth.shape)
        H, W = f.depth.shape
        print(f"  peak indentation at index {idx} of shape {f.depth.shape}")
        print(f"  If that is (small, small): axis0 rows match config field.h={hw.tactile.field.h}"
              f" -> hw_order: hw")
        print(f"  If the axes look swapped vs the physical press -> hw_order: wh")
    finally:
        s.disconnect()


def bench_a_units(hw) -> None:
    from phantom.drivers.real.dmtac import DmTacSensor
    print("\n=== (a) force units ===")
    print("  Place a known calibration mass M (grams) on the sensor face, enter it.")
    mass_g = float(input("  mass in grams: "))
    expected_N = mass_g / 1000.0 * 9.81
    s = DmTacSensor(hw.tactile, hw.tactile.sensors[0])
    s.connect()
    try:
        vals = []
        for _ in range(60):
            f = s.read()
            vals.append((f.wrench[2], f.dist_force[..., 2].sum()))
        wz = float(np.median([v[0] for v in vals]))
        dz = float(np.median([v[1] for v in vals]))
        print(f"  expected normal force: {expected_N:.3f} N")
        print(f"  getForce z median: {wz:.4f} (SDK units)"
              f" -> tactile.force_unit_to_N = {expected_N / wz if wz else 0:.5f}")
        print(f"  sum(getDistributeForce z): {dz:.4f}"
              f" -> tactile.dist_force_unit_to_N = {expected_N / dz if dz else 0:.6f}")
        print("  If irreproducible/nonlinear: leave both at 0.0 (deformation fallback)")
    finally:
        s.disconnect()


def bench_d_throughput(hw) -> None:
    print("\n=== (d) recording throughput ===")
    est = hw.field_bytes_per_s() * hw.n_fingers / 1e6
    print(f"  config-derived estimate: {est:.1f} MB/s pre-compression")
    print("  Run a 60 s mock->real recording and watch the recorder watchdog:")
    print("    python -m phantom.scripts.mock_smoke --seconds 60   (with real tactile"
          " via mode.overrides)")


def bench_arm(hw) -> None:
    from phantom.drivers.real.ur import URArm
    print("\n=== arm generation / wrist F/T ===")
    arm = URArm(hw)
    arm.connect(control=False)
    try:
        n, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < 3.0:
            st = arm.get_state()
            n += 1
        rate = n / (time.perf_counter() - t0)
        print(f"  RTDE receive sustained: {rate:.0f} Hz "
              f"(configured {hw.arm.rtde_receive_hz})")
        print(f"  ft sample: {st.ft.round(3)}  robot_mode={st.robot_mode}")
        print("  e-Series (PolyScope 5.x / built-in F/T): arm.generation: e-series,"
              " wrist_ft.source: ur_internal, rate 500")
        print("  CB3: arm.generation: cb3, order an FT-300S, wrist_ft.source: ft300s,"
              " rate 100, rtde <= 125 Hz")
    finally:
        arm.disconnect()


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", nargs="*", default=["b", "c", "a", "d", "arm"])
    ap.add_argument("--hardware", default=None)
    args = ap.parse_args(argv)
    hw = load_hardware(args.hardware, quiet=True)
    fns = {"a": bench_a_units, "b": bench_b_shapes_rates, "c": bench_c_ordering,
           "d": bench_d_throughput, "arm": bench_arm}
    for item in args.items:
        try:
            fns[item](hw)
        except Exception as e:
            print(f"  [{item}] FAILED: {e}")
    print("\nUpdate configs/hardware.yaml with the values above, rerun the failed "
          "items, then set meta.bench_verified: true and rerun mock_smoke + pytest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
