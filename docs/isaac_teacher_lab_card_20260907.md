# Teacher lab card — final v5 results

**Next lab candidate: ftA1500, teacher EMA, NFE1, K4, guidance 1.** This is a conditional recommendation, not a validated reliable winner.

| Reserved confirmation only | Acquired | Lifted | Carried | Full placement | Clean finish | Drops |
|---|---:|---:|---:|---:|---:|---:|
| ftA1500 / NFE1 / K4 |12/12|7/12|6/12|1/12|1/12|1/12|
| ftA3000 / NFE1 / K4 |8/12|4/12|4/12|0/12|0/12|0/12|

The physical-placement difference is +8.33 percentage points, paired 95% interval [0,+25], exact McNemar p=1. The predeclared winner gates fail. Screening, older studies, and development successes are not pooled into these counts. [Independent final audit](results/teacher_success_anchor_v5/confirmation/independent_audit.md).

Use `/home/physicalai/phantom-icra-2027/phantom/runs/teacher_v5_ftA/teacher_001500.pt`, SHA256 `67c93287123e447b85bf32385660f1a99d6a8b0ad5e995727ac92a680543893e`. Keep teacher architecture, EMA, task/text `waffles`, persistent noise, parity, and maximum 10 played actions. K4 means four candidate chunks per replan; it is separate from the episode seed.

The native controller to qualify is the explicit **`--placement-controller-profile minimal_v5`** port in [PR14](https://github.com/Advanced-Robotic-Manipulation/phantom/pull/14), introduced by commit `51457a36a8f3af3e4dbdc0100a8d6121aae70804`. It retains historical request-time veto and uses live real sensors. It requires `--terminal-veto` and an explicitly measured `--placement-release-config` with FINISH enabled. It has CPU parity evidence, not hardware validation or automatic installation. See [exact configuration and operator handoff](isaac_lab_handoff_20260907.md).

For the next bench session:

1. Measure base/table datum, pad/TCP faces and backing, actual aperture, packet/box dimensions and reset marks, camera alignment, and unloaded/known-load sensor response. Use the [measurement guide](isaac_teacher_measurements_20260907.md) and [sheet](measurements/setup_20260907_template.csv).
2. Qualify attended pickup/retention and the measured release/FINISH volume on one reviewed start. Keep existing reach, speed, load, freshness and start-distribution guards.
3. Fix the waffle, box, camera and lighting. Then vary authentic measured arm/gripper starts in a declared 12-case pilot: six reviewed starts × two fresh seeds, with all attempts retained. The policy generates the trajectory. Object-reset jitter is a later separate factor; do not randomize both factors together.

The successful illustration physically placed at 24.268 s, finished controller release at 25.076 s, and remained unloaded in the box through 60 s. Its final aperture also includes historical recovery after physical placement; this is documented in the [event audit](results/teacher_success_anchor_v5/selected_success_audit.md).

Rigid packet/compliance, gel mapping and the pad-only wrist proxy remain approximate. A failed approach exceeded 120 N in simulated contact while the rolling wrist reference absorbed gradual load; that is a realism/guard limitation, not evidence of safe force transfer. [Contact audit](results/teacher_success_anchor_v5/contact_force_audit.md). The separate V6 limiter stopped both cases before release and is not promoted. The completed [V7 settings diagnostic](results/teacher_success_anchor_v7/README.md) supports retaining K4: it lifted2/2 and placed1/2; faster K1 lifted0/2. Those development results are not pooled into confirmation. No hardware was run or changed by this work.
