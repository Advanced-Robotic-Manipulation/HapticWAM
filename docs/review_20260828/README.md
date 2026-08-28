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
