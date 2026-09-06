# Fresh August 22 mapper validation

**Passed the software and observer-invariance checks. This is a measured-motion replay, not a policy trial or hardware calibration.** The fresh 297-frame replay spans 0–19.732 s. Every one of its 21 state/contact arrays is bitwise identical (including NaNs) to the previously validated `measured_dynamics_support` replay. The effective scene configuration, initial state file, camera intrinsics, robot asset path and all 13 robot / 5 bin support filters also match.

The [numeric audit](mapper_fresh_replay_audit.json) records input hashes, every array comparison, same-contact v1/v2 scalar readouts and representative differences. Its frozen mapper SHA is `d7f4cb1d4606aa0ced6d360be6073498e5505460248d75cf6d04c7cf52853d6e`.

## Actual contact evidence

The 594 pad readouts contain 864 unique populated contacts: 778 have positive normal-force magnitude and 86 have zero impulse. No duplicate filter indices occur. All populated signed separations are present and finite; they range from −5.468 to +1.971 mm. Loaded contacts reach at most +0.0177 mm separation. Zero-impulse vertices have positive gaps of +0.0842 to +1.971 mm.

V2 routes 195 body readouts through the declared single-convex `/World/Waffle` patch calculation; 17 of those patches use zero-impulse support geometry. There are 40 explicit point fallbacks: 12 lack eligible positive load, 7 lack three noncollinear geometric vertices, and 21 belong to undeclared bodies (13 mat and 8 bench readouts). No geometry exceeds the fixed 2 mm contact-generation envelope. The maximum populated contacts in one pad readout is 10, far below its 256-entry capacity.

Recomputing V2 from the archived populated indices, points, normals and separations reproduces every saved force and UV exactly. The largest normal-force conservation error is 4.45×10⁻¹⁶ N; the projection budget error is 8.89×10⁻¹⁶ N and the ignored-reason budget error is 3.56×10⁻¹⁵ N. No load is created or clipped.

## What changed from V1

Seven of 594 readouts change by more than 10⁻⁸ N. The largest change is right-pad gel force at 7.400 s: V1 gives 0 N and V2 gives 3.406 N. This readout has one loaded Waffle constraint and three populated zero-impulse geometric vertices. Its fixed convex support overlaps 98.195% of the active ellipse. The packet contributes 3.708 N raw normal magnitude; projection and area weighting yield 3.406 N gel compression. A separate 33.001 N mat load remains fully rejected by the gel model and remains a physical contact load.

Peak gel forces change from V1 `[2.964, 3.103]` N to V2 `[2.964, 3.406]` N. Both versions have exactly 73 scene samples with bilateral gel force at least 2.5 N. These scene-rate counts are not the policy's causal 8 Hz latch history. They do not establish a changed policy outcome.

The fixed motion and physical support chain are unchanged: the reference scorer records acquired, sustained lift, carry and bin-supported release, with full-chain confirmation at 14.268 s. Its only invalid-scoring reason is `run_mode_is_not_policy`. That invalidity is expected and remains in the audit; replay success is not counted as policy success.

## Remaining physical limits

The deepest reported penetration occurs at 6.736 s: a left-pad/Waffle constraint has −5.468 mm signed separation and 38.953 N normal load. The point lies near the inner X position, but its normal is predominantly tangent to the gel; it remains excluded from the gel image. The large penetration and load require the separate timestep/solver/contact-model convergence check. The mapper fix does not resolve that physical issue.

A positive PhysX separation is a speculative contact gap within the sum of shape contact offsets; it is not measured gel indentation. This convention follows the [PhysX contact-point API](https://nvidia-omniverse.github.io/PhysX/physx/5.1.0/_build/physx/latest/struct_px_contact_pair_point.html) and [contact-offset documentation](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/AdvancedCollisionDetection.html). The scene authors 1 mm contact offset and zero rest offset on each pad and packet collider. V2's geometry gate therefore uses their fixed 2 mm sum, independently of impulse allocation.

Uniform pressure over the support polygon is still an explicit approximation. The tensor API does not expose a calibrated pressure image or separate manifold-patch/collider identities. The convex-body whitelist and normal-coherence check reduce unsupported interpolation, but cannot prove real connected gel loading. Friction, shear, SDK image orientation, packet compliance and real contact area remain uncalibrated.

## Reproduce without simulation

From the repository root:

```sh
ssh compute3 'PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 nice -n 19 /home/physicalai/phantom-icra-2027/phantom/.venv/bin/python - --fresh /home/physicalai/phantom-icra-2027/sim/waffles/runs/teacher_v2_preflights/dynamics_aug22_mapper_v2 --reference /home/physicalai/phantom-icra-2027/sim/waffles/runs/teacher_pick_place_v1/diagnostics/measured_dynamics_support --source /home/physicalai/phantom-icra-2027/sim/waffles/source_teacher_v2_mapper_preflight1' \
  < docs/results/teacher_v2_design/audit_mapper_fresh_replay.py \
  > /tmp/mapper_fresh_replay_audit.json
```

The [CPU helper](audit_mapper_fresh_replay.py) reads immutable numeric records only; it does not launch Isaac or modify source records. Ruff checks pass.
