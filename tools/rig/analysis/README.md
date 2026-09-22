Read-only analysis of real deploy episodes (run on the rig box, `phantom/.venv/bin/python <script>`).

- `apex_analysis.py <episodes_root> <out.json>` — per-episode apex/reach/stop table grouped by task and checkpoint (writes the JSON the other scripts read from `/tmp/apex_0911.json`).
- `apex_detail.py` — post-closure (t, z, r_h, wd) samples for lifted waffles episodes.
- `wrist_geom.py` — elbow/wrist positions from `phantom.sim.kinematics.dh_frames` at apex and max extension, demos vs policy.
- `zcap_counterfactual.py` — IK re-solve of recorded trajectories with the carry z capped.
- `limiter_replay.py` — replays recorded trajectories through `servo_limiter.select_servo_step` (bounded_v1 limits), with/without a 0.35 m ceiling.
- `wall_cross.py` — TCP height when crossing y=0/+0.03/+0.06 (over the box) and end force.
- `old_demos_tcp.py` — carry/crossing geometry over all waffles demo sessions (TCP only; old demos lack joints).
