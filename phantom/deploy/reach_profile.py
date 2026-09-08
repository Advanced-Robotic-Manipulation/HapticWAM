"""Explicit native reach-limiter configuration; no drivers or model imports.

``bounded_v1`` activates the existing shared servo limiter and its verified
constraint-hold budget. It does not change measured safety stops, task bounds,
gripper behavior, inference, or placement/release logic. This constrains
submitted joints; it does not guarantee measured tracking or task completion.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from phantom.config.hardware import HardwareConfig, load_hardware

PROFILE = "bounded_v1"
PROFILE_LIMITS = {
    "elbow_min_rad": 0.40,
    "servo_joint_speed_max_rad_s": 1.0,
    "servo_constraint_hold_s": 2.5,
}
_UR3_DH = (0.24365, 0.21325, 0.11235)
_UR3_SHOULDER_M = 0.1519


def apply_reach_profile(hw: HardwareConfig, profile: str | None) -> HardwareConfig:
    """Return a validated copy, or the original object when not selected.

    This named package supports only the configured six-joint UR3 DH
    geometry. Existing non-default limits must agree, so selection cannot
    silently replace a more conservative custom package. Generation labels
    are preserved: the lab's CB3 uses an e-series config label for its current
    wrist-force interface workaround; this helper cannot identify hardware.
    """
    if profile is None:
        return hw
    if profile != PROFILE:
        raise ValueError(f"Unknown servo reach profile: {profile}")
    if hw.arm.model.strip().upper() != "UR3" or hw.arm.dof != 6:
        raise ValueError("bounded_v1 requires the configured six-joint UR3")
    safety = hw.safety
    geometry = (*safety.ur_dh_a2_a3_d4_m, safety.ur_dh_d1_m)
    if not all(math.isclose(a, b, rel_tol=0, abs_tol=1e-9)
               for a, b in zip(geometry, (*_UR3_DH, _UR3_SHOULDER_M))):
        raise ValueError("bounded_v1 requires the verified UR3 DH geometry")
    a2, a3, d4 = _UR3_DH
    limited_radius = math.sqrt(a2*a2 + a3*a3 + 2*a2*a3*math.cos(0.40) + d4*d4)
    full_radius = math.sqrt((a2 + a3)**2 + d4*d4)
    stop = safety.wrist_extension_stop_m
    if stop is None or not math.isfinite(stop) or not limited_radius < stop < full_radius:
        raise ValueError(
            "bounded_v1 requires an enabled wrist-extension stop outside its "
            f"{limited_radius:.6f} m command boundary and inside full UR3 reach; "
            "the profile does not change that measured guard"
        )
    speed_stop = safety.joint_speed_stop_rad_s
    if not math.isfinite(speed_stop) or speed_stop < PROFILE_LIMITS["servo_joint_speed_max_rad_s"]:
        raise ValueError("bounded_v1 joint command cap exceeds the configured measured speed stop")
    for name, value in PROFILE_LIMITS.items():
        current = getattr(safety, name)
        if current is not None and current != value:
            raise ValueError(
                f"bounded_v1 conflicts with safety.{name}={current}; "
                f"expected unset or {value}. No configured limit was changed."
            )
    data = hw.model_dump(mode="python")
    data["safety"].update(PROFILE_LIMITS)
    return HardwareConfig.model_validate(data)


def main(argv=None) -> int:
    """File-only configuration check: never import or construct drivers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", type=Path, required=True)
    parser.add_argument("--profile", choices=[PROFILE], required=True)
    args = parser.parse_args(argv)
    try:
        base = load_hardware(args.hardware, quiet=True)
        effective = apply_reach_profile(base, args.profile)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({
        "check": "configuration_only_no_hardware",
        "profile": args.profile,
        "hardware": str(args.hardware.resolve()),
        "base_config_hash": base.config_hash(),
        "effective_config_hash": effective.config_hash(),
        "effective_limits": {name: getattr(effective.safety, name) for name in PROFILE_LIMITS},
        "effective_hardware": effective.model_dump(mode="json"),
    }, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
