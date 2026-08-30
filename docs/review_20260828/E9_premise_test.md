# E9 — does the teacher's action head actually use tactile? (2026-08-29/30)

`tools/terminal_eval.py` (hardened: all episodes, medians, `pred_close_height_mm`, `--null`) on v5_6 (EMA),
val split of `data/val_eval/tasks` on compute3 (78 episodes × 4 seeds = 312 terminal windows, NFE 5, guidance 1.0).

    .venv/bin/python tools/terminal_eval.py --ckpt runs/teacher_v5_batch0822/v5_6.pt \
        --data ~/phantom-icra-2027/data/val_eval/tasks --hardware configs/hardware.nuc.yaml \
        --seeds 4 --null {none,tactile,wrist,prev_cpk,contact_gt} --out /tmp/e9_<mode>.json

| condition | endpoint err mm | z-at-end err mm | commit ratio | close step err | pred close height mm |
|---|---|---|---|---|---|
| real inputs | 20.86 | 8.53 | 0.94 | −0.60 | 110.8 |
| tactile nulled | 23.24 | 11.64 | **0.79** | **−1.93** | 116.2 |
| wrist F/T nulled | 21.06 | 8.60 | 0.95 | −0.78 | 110.9 |
| prev_cpk nulled | 20.86 | 8.54 | 0.94 | −0.64 | 110.8 |
| CONTACT frames zeroed (no GT package) | 23.96 | 14.12 | **0.65** | **−2.44** | 120.9 |

Per task (endpoint err mm / close step err):

| condition | Carton | egg | waffles | whiteboard |
|---|---|---|---|---|
| real | 25.8 / +0.3 | 7.6 / −3.5 | 29.5 / +0.2 | 18.6 / −0.1 |
| tactile nulled | 27.9 / −0.3 | 7.9 / −4.9 | 33.1 / −0.4 | 21.8 / −2.6 |
| wrist nulled | 26.2 / +0.2 | 7.6 / −3.8 | 29.7 / +0.1 | 18.8 / −0.1 |
| prev_cpk nulled | 25.8 / +0.3 | 7.6 / −3.6 | 29.6 / +0.2 | 18.6 / −0.1 |
| CONTACT zeroed | 27.9 / −0.9 | 8.4 / −4.8 | 32.7 / −0.4 | 24.5 / −4.0 |

Reading:
- **The premise holds, modestly.** Removing the live tactile input costs 16% of terminal commitment and pulls the
  close ~1.3 steps earlier on every task (Δ endpoint 2.4 mm — at the review's 3 mm "trouble" line, but commit and
  close timing are well past their thresholds). Wrist F/T and prev_cpk are inert (consistent with E3).
- **The co-denoised GT contact package carries MORE of the terminal commit than the live tactile input**
  (commit 0.94 → 0.65 when the CONTACT frames are zeroed). At deploy those frames are the model's own imagination,
  never GT — this is P7 (exposure bias) measured, and the direct argument for `--contact-self-forcing` in FT-A.
- Both effects are offline, on demo states; the rig ablation (`--terminal-veto` on/off, tactile on/off) decides
  what they are worth in closed loop.
