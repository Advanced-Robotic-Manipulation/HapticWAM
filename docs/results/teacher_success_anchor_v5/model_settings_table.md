# Frozen v5 teacher settings for the final handoff

**Next lab candidate: `fta1500_nfe1_k4`; no validated reliable winner.**
Reserved confirmation gave ftA1500 **12/12 acquisitions, 7/12 lifts, 6/12 carries,
1/12 strict clean placements and 1/12 drops**; ftA3000 gave **8/12 acquisitions,
4/12 lifts/carries, 0/12 strict placements and 0/12 drops**. Screening and older
campaigns are not pooled. Paired physical-placement difference is +8.33 percentage
points, 95% interval [0,+25], exact McNemar p=1; the predeclared winner gates fail.
See the [independent confirmation audit](confirmation/independent_audit.md).

The table retains the four frozen tested recipes. Recommendation means the next
attended calibration/transfer candidate, not a physical success guarantee.

| Candidate | NFE | K | Guidance | Parity fixes | Persistent noise |
|---|---:|---:|---:|---|---|
| `fta1500_nfe1_k4` | 1 | 4 | 1 | enabled | enabled |
| `fta3000_nfe1_k4` | 1 | 4 | 1 | enabled | enabled |
| `v5_6_nfe1_k4` | 1 | 4 | 1 | enabled | enabled |
| `fta1500_nfe5_k4` | 5 | 4 | 1 | enabled | enabled |

These are the five inference controls. Native flag names are `--nfe`,
`--k-seeds`, `--guidance`, `--parity-fixes` and `--persistent-noise`.
All rows separately require **teacher architecture, EMA weights**, task text
`waffles` and maximum playback of **10** actions. Native corresponding selectors
include `--ema`, `--task waffles` and `--max-play-steps 10`. These are setting
fragments, not a complete hardware control command. Do not substitute the older
NFE5/K1 diagnostic for the declared **NFE5/K4** candidate.

| Checkpoint | Full SHA256 |
|---|---|
| `/home/physicalai/phantom-icra-2027/phantom/runs/teacher_v5_ftA/teacher_001500.pt` | `67c93287123e447b85bf32385660f1a99d6a8b0ad5e995727ac92a680543893e` |
| `/home/physicalai/phantom-icra-2027/sim/waffles/checkpoints/teacher_v5_ftA/teacher_003000.pt` | `ee0a448c00da2fe047b4bcba5036a8b79e9ae33e3a6ec47ea69d27466b4ff1dc` |
| `/home/physicalai/phantom-icra-2027/phantom/runs/teacher_v5_batch0822/v5_6.pt` | `7edcb8335681e19bead5ad39a2a80fe3fd8c5893b1761d2dd812c304881fe65c` |

The exact paths, original setting dictionaries and flag fragments are in
[model_settings_table.json](model_settings_table.json). They retain each payload's
own architecture, normalizers and checkpoint-specific noise configuration;
there is no cross-checkpoint normalizer or noise-shape substitution.

Source configuration:
`configs/sim/teacher_success_anchor_v5_screen.json`, SHA256
`3392e9f703ef008f2a16d6c133870192ddb1642526ef7adb3702b2e8a6c9b9c5`.
Parent protocol SHA256:
`f67fe79a5267c6d253d7d90f67f2a367c8899f030959c8f42fa714d072ab7ac2`.

The completed confirmation is hash-bound in the JSON. Formal `selected_id`
remains null because `clear_simulator_winner` is null. The separate
`recommended_lab_candidate_id` records the conditional ftA1500 lab choice;
it does not relabel that model as a validated winner or enable the winner-only
simulator arm extension. [Final lab handoff](../../isaac_lab_handoff_20260907.md).

Simulator source/profile remains immutable minimal v5 (historical request-feedback
veto, gel coverage v2, original pad-contact wrist and FINISH). The separately
[qualified native controller port](native_controller_port.md) consumes live real
sensors and still needs geometry, timing and physical release checks. Neither
the simulator proxy parameters nor the source configuration are a ready-to-run
physical robot configuration.
