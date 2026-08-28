# Full review + literature scan, 2026-08-28

Produced after rig session #3 (0/26 successes). Read `REVIEW_SYNTHESIS.md` first: ten confirmed
problems (P1-P10), ranked root-cause hypotheses (H1-H7), the offline experiment table (E0-E9),
a day-by-day plan to the ICRA deadline with gates (G0-G5), the cut order, 14 refuted findings
(do not re-raise), and the completeness critic.

- `lenses/` — the eight Opus review lenses (data pipeline, model+losses, training loop, train/deploy
  parity, closed-loop root cause, paper claims, deploy safety, self-improvement readiness) + probe scripts.
- `codex/` — independent Codex (gpt-5.4) passes per lens.
- `research/` — six literature reports (closed-loop chunk execution, covariate shift / DAgger,
  real-robot RL on flow policies, tactile terminal control, world-action models + distillation,
  failure demos + tactile success reward); `CONTEXT.md` is the brief they worked from.

Status when saved: the floor-vs-hitbox safety bug (§1.11) is fixed (commit f8e18bb). Nothing else
from the plan is implemented yet; D1 items (replay_rig.py, P9 label guard, P10 --student / mc-from-
checkpoint, parity bundle) are next.

`rig_0828/` — the session-#3 evidence: per-episode commanded-vs-actual tables (`episodes_0828_commanded_vs_actual.txt`,
from tools/rig_trace_decompose.py), start joint configuration vs demos + demo close statistics
(`joint_ood_and_demo_close_stats.txt`), scene frames early vs late session (`scene_frames_early_vs_late.jpg`:
rows 3-6 show the flipped/wrapped arm configuration and the waffles pack used in the "Carton" runs), and the
one-off analysis scripts. Operator reports from the session (Petr): a waffles-pack/table slam, an electric shock
from the arm after the E-stop (hardware/earth issue, not software), gripper stuck closed after a pad-on-pad
close (→ gripper_ctl), request for a no-go zone + speed limits (→ z floor, hitbox, --max-tcp-speed).
`workflow_phantom-full-review.js` — the review workflow script (8 lenses × Opus+Codex → 2-vote verification → synthesis).
