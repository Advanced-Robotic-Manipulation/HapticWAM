# Recovery-demo collection protocol (teacher v5 fine-tune)

**Why.** Rig session 2026-08-20: the v4 teacher approaches correctly, then drifts
3–8 cm off / too high during the final descent and closes on air (13/13
episodes). Offline it executes the terminal phase correctly *from demo states*
— it has never seen how to recover from slightly-wrong states, because every
demo started from a good one. These demos teach exactly that.

**What to record.** Normal teleop demos with the standard pipeline (same rig,
same camera, same upload) — except each episode **starts from a deliberately
wrong pre-grasp state** and the operator performs the correct recovery + grasp
+ place.

## Per episode

1. Place the object per the task's usual placement (vary position ±5 cm and
   rotation ±20° across episodes — do NOT put it in the same spot every time).
2. Jog the arm to a **wrong hover pose** above/near the object, then start
   recording. Draw the offset from this menu (roughly equal counts of each):
   - 3–8 cm lateral in a random direction (left/right/toward/away from base)
   - 3–8 cm too high, straight above
   - lateral + high combined
   - 1 in 5 episodes: *already descended too far beside the object* (gripper at
     grasp height but 3–6 cm off) — the most common rig end-state
3. Gripper at the task-typical opening (waffles ~0.2–0.35, Carton ~0.1, egg
   ~0.3, whiteboard ~0.3). Not half-closed.
4. Teleop the **correction first** (translate over the object, descend), then
   grasp and complete the task as in a normal demo. Smooth, no hesitation at
   the hover — the point is "from here, commit".
5. Mark success/failure honestly at the end. A genuine failed recovery is
   still useful data (label it failed).

## Quantities (minimum)
| task | recovery demos |
|---|---|
| waffles | 20 |
| whiteboard | 15 |
| Carton | 15 |
| egg | 10 |

≈ 60 episodes, one session. More is better; placement variety matters more
than count beyond this.

## Do / don't
- DO keep lighting as in the original demos (same lamps on).
- DO keep hands out of the camera view during the episode.
- DON'T start from an absurd pose (>10 cm off, wrong side of the bin) — the
  rig never ends up there; keep offsets in the 3–8 cm band.
- DON'T "fix" the object position by hand mid-episode.

## After the session
Episodes upload to the hub automatically. Ping Mikhail/Claude: the fine-tune
(`train_teacher --init-weights teacher_020000.pt` on demos + recoveries,
photometric augmentation on) runs on compute in ~6 h, offline-checked with
`tools/terminal_eval.py`, then staged as `DEMO.pt` for the next rig session.
