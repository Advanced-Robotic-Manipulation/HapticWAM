# Recorded arm-start preflight

All ten native August22 start states pass the existing two-second free-hold equilibrium checks. No start was excluded. The canonical green-waffle pose, source initial-state SHA and requested native joint values match exactly for every start.

The [CPU audit](start_preflight_audit.json) also applies proposed geometry-only checks of packet drift≤2mm and end-of-settling packet/robot normal load≤0.1N. Every start passes with packet drift1.4323×10⁻⁷m and zero robot/packet load. These checks were proposed before reading policy outcomes; they are a preflight extension, not a changed policy-success threshold.

| Recording | Maximum joint error (rad) | Final maximum joint speed (rad/s) | Minimum modeled gripper/environment vertical clearance (mm) |
|---|---:|---:|---:|
| ep_waffles_1787395928_000 | 0.003670 | 0.029147 | 139.95 |
| ep_waffles_1787395963_001 | 0.003452 | 0.029805 | 134.67 |
| ep_waffles_1787396028_003 | 0.003577 | 0.029959 | 122.86 |
| ep_waffles_1787396060_004 | 0.003600 | 0.029972 | 138.52 |
| ep_waffles_1787396094_005 | 0.003641 | 0.029328 | 95.19 |
| ep_waffles_1787396128_006 | 0.003582 | 0.029914 | 114.91 |
| ep_waffles_1787396273_010 | 0.003640 | 0.029090 | 82.31 |
| ep_waffles_1787396314_011 | 0.003605 | 0.028698 | 143.08 |
| ep_waffles_1787396346_012 | 0.003571 | 0.029278 | 126.80 |
| ep_waffles_1787396461_000 | 0.003675 | 0.029006 | 107.68 |

The enforced runtime limits remain joint error≤.02rad and speed≤.05rad/s. The source file preserves the actual measured q/gripper/wrist samples; it does not replace an inconvenient start with a later pose. Initial velocity is separately set and allowed to settle in physics. No policy inference was run in these save-stage preflights.

The analytical gripper-clearance proof includes the configured elliptical pads, backing, linkage and box housing; all are above every packet/table/mat/bin top by at least the stated bound. The nominal UR3 FK differs from native synchronized TCP by1.416–1.463mm. That common calibration residual is reported without inventing a joint-angle exclusion.

**Limits:** these records do not certify all UR3 self-collision or arm/environment pairs. A stationary arm can carry contact force without moving, so equilibrium alone is not a universal collision test. The saved initialization includes only final packet contact/support; it cannot rule out transient packet contact earlier in settling. `--save-stage-only` returns before writing run.json/sim_trace, so those files are intentionally absent. The exported stages and `robot_settling.npz` remain available on compute3 for inspection.

Reproduce using [audit_start_preflight.py](audit_start_preflight.py), with `--source /home/physicalai/phantom-icra-2027/sim/waffles/source_teacher_v2_mechanics1 --runs /home/physicalai/phantom-icra-2027/sim/waffles/runs/teacher_v2_preflights` on compute3. The helper is read-only and CPU-only.
