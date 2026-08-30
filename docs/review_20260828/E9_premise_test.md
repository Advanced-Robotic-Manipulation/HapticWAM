# E9 — does the teacher's action head actually use tactile? (2026-08-29/30, RE-RUN and RELABELLED 2026-08-30)

> **The 2026-08-29 table was read backwards and one row was mislabelled.** Its row 5,
> "CONTACT frames zeroed (no GT package)", was in fact the `--null contact_gt` run: the GT contact
> package KEPT and its CONTACT frames cond-PINNED to it. The default condition (`--null none`)
> already zeroes the GT package — `terminal_eval` zeroes `events` and every `cpk_*` key in every mode
> but `contact_gt` — so "no GT package" was never a separate row, and the conclusion drawn from it
> ("the co-denoised GT contact package carries MORE of the terminal commit", i.e. exposure bias, i.e.
> the argument for `--contact-self-forcing` in FT-A) has the sign of the effect inverted.
> Re-run below with the fixed tool, plus the control arm P7 actually asked for.

`tools/terminal_eval.py` on v5_6 (EMA), val split of `data/val_eval/tasks` on compute3
(78 episodes × 4 seeds = 312 terminal windows, NFE 5, guidance 1.0, `skipped: 0`).

    .venv/bin/python tools/terminal_eval.py --ckpt runs/teacher_v5_batch0822/v5_6.pt \
        --data ~/phantom-icra-2027/data/val_eval/tasks --hardware configs/hardware.nuc.yaml \
        --seeds 4 --null {none,contact_zero,contact_gt,tactile} --out /tmp/e9_<mode>.json

Every run now prints and records `summary["null_semantics"]`, so no table built from these files can
be relabelled by hand again. The two switches that define a condition:

| `--null` | GT contact package in the batch | CONTACT frames during denoising |
|---|---|---|
| `none` (default) | **zeroed** | co-denoised from noise — **this is the deploy condition** |
| `tactile` | zeroed | co-denoised (and gel / fields / contact_state / **reactive** zeroed) |
| `wrist`, `prev_cpk` | zeroed | co-denoised |
| `contact_zero` *(new)* | zeroed | **cond-pinned to the zero package** |
| `contact_gt` | **kept** | **cond-pinned to the GT package** |

## The table

| condition | endpoint err mm | median | z-at-end err mm | commit ratio | close step err | pred close height mm |
|---|---|---|---|---|---|---|
| **`none` — real inputs, no GT package (deploy)** | 20.86 | 17.99 | 8.53 | **0.94** | −0.60 | 110.8 |
| `tactile` — gel/fields/contact_state/reactive zeroed | 23.24 | 20.14 | 11.64 | **0.79** | −1.93 | 116.2 |
| `contact_zero` — CONTACT frames PINNED to zeros | 25.29 | 23.42 | 16.75 | **0.58** | **+1.24** | 98.7 |
| `contact_gt` — GT package KEPT and PINNED | 23.96 | 23.85 | 14.12 | **0.65** | **−2.44** | 120.9 |

GT close height (a property of the demos, identical in every condition): 103.8 mm.
`wrist` (21.06 / 0.95 / −0.78) and `prev_cpk` (20.86 / 0.94 / −0.64) are unchanged from 2026-08-29 —
inert, consistent with E3 — and were not re-run; their `--null` handling did not change.

Per task (endpoint err mm / close step err / commit ratio):

| condition | Carton | egg | waffles | whiteboard |
|---|---|---|---|---|
| `none` | 25.8 / +0.3 / 1.46 | 7.6 / −3.5 / 0.88 | 29.5 / +0.2 / 0.51 | 18.6 / −0.1 / 0.89 |
| `tactile` | 27.9 / −0.3 / 1.24 | 7.9 / −4.9 / 1.05 | 33.1 / −0.4 / 0.41 | 21.8 / −2.6 / 0.59 |
| `contact_zero` | 31.5 / +1.6 / 0.80 | 8.7 / −1.7 / 0.79 | 35.2 / +1.4 / 0.23 | 23.5 / +3.1 / 0.60 |
| `contact_gt` | 27.9 / −0.9 / 0.90 | 8.4 / −4.8 / 0.72 | 32.7 / −0.4 / 0.33 | 24.5 / −4.0 / 0.66 |

## Reading

- **What the default condition contains.** `--null none` is real (teacher-forced demo) observations
  with **no GT contact package**: `events` and all `cpk_*` are zeroed and the CONTACT frames are
  co-denoised from noise together with the actions — exactly what deploy does. It is the reference
  row, not an "everything on" row. The steady-state `prev_cpk` handed to the terminal window is the
  model's OWN package sampled one chunk earlier, never GT.

- **The tactile premise holds, modestly — unchanged.** Removing the live tactile streams costs 16%
  of terminal commitment (0.94 → 0.79) and pulls the close ~1.3 steps earlier on every task, for
  Δ endpoint 2.4 mm — at the review's 3 mm "trouble" line, with commit and close timing well past
  their thresholds. `--null tactile` now also zeroes `reactive` (F12: it is
  `derived.reactive_score` of two consecutive `fields_ds` frames, i.e. tactile, and was surviving
  the "tactile nulled" condition), and the re-run is **bit-identical to 2026-08-29 in all 312
  windows** — `reactive` reaches only ACC's `psi_react` → `g_react` → gate `g` (`model/acc.py:95-103`),
  which is diagnostic at sampling time and never enters the action denoiser. The mode is now honest
  about what it removes; the number it produces did not depend on it.

- **Which direction GT-pinning moves commit: DOWN.** Handing the model the true future contact
  package and holding it fixed through every denoise step makes it commit **less** (0.94 → 0.65,
  −31%), close 1.8 steps **earlier** and 10 mm **higher** (110.8 → 120.9 mm), and land 3.1 mm worse
  at the endpoint. The published sentence — "the co-denoised GT contact package carries MORE of the
  terminal commit than the live tactile input" — states the opposite of the measurement, and the
  commit message ("zeroed GT contact package costs 31%") attributes the 31% to a condition that was
  never run.

- **Most of that drop is the PINNING, not the package's content.** The new control arm answers this:
  pinning the CONTACT frames to the ZERO package — the same package the default already has — drops
  commit further, to 0.58, and worsens the endpoint to 25.3 mm. Both pinned arms are far below the
  free-running default; what the pinned CONTENT changes is the *direction* of the close error — GT
  closes 2.4 steps early at 120.9 mm, 17 mm ABOVE the demos' own 103.8 mm close height, while zeros
  close 1.2 steps late at 98.7 mm, 5 mm BELOW it (the default sits at 110.8 mm, 7 mm above). The
  honest description is that cond-pinning the CONTACT frames is itself off-distribution — training
  co-denoises those frames as targets and never holds them fixed — so neither pinned number can be
  read as "the head reads the answer off GT contact tokens".

- **What this implies for `--contact-self-forcing`: E9 does not support it.** P7's exposure-bias
  argument needs the model to do BETTER when handed GT contact than when imagining it, so that
  training on its own predictions would close a measured gap. The deploy condition is the best of the
  three on every metric (endpoint 20.9 vs 24.0 / 25.3; commit 0.94 vs 0.65 / 0.58; close error −0.60
  vs −2.44 / +1.24). There is no measured GT-vs-imagined gap of the P7 shape to close, so **the E9
  row cannot be quoted as the justification for `--contact-self-forcing` on a paid H100 fine-tune**;
  the flag needs an independent argument or stays off. The FT-A A/B is now the only thing that can
  decide it.

- Both effects are offline, on demo states; the rig ablation (`--terminal-veto` on/off, tactile
  on/off) decides what they are worth in closed loop.

## Provenance

- Fixed tool: `tools/terminal_eval.py` at branch `fixnow/eval-0830` (F12: `student=mc.student`,
  `reactive` in `--null tactile`; F18: `--null contact_zero`, `null_semantics` in the JSON and in
  `--help`). Copied to compute3 `/tmp/te_e9_fixed.py` for the run; the checked-out repo there was not
  modified.
- Re-run JSONs on compute3: `/tmp/e9_{none,contact_zero,contact_gt,tactile}.json` (2026-08-30
  21:31–22:35 MSK, one mode at a time, GPU shared with another job for the last mode).
  The 2026-08-29 originals are preserved in `/tmp/e9_orig_0829/` (`wrist` and `prev_cpk` there are
  the rows quoted above; `none`/`tactile`/`contact_gt` there are superseded by the re-run).
- `none` reproduces the 2026-08-29 "real inputs" row to the last digit (20.86 / 8.53 / 0.94 / −0.60 /
  110.8) and `contact_gt` reproduces its row 5 (23.96 / 14.12 / 0.65 / −2.44 / 120.9) — which is the
  arithmetic proof that row 5 was the GT-PINNED run all along, not a zeroed-package run.
