# PHANTOM — research brief for the literature agents (2026-08-28)

## The paper (ICRA 2027, deadline ~Sep 15 2026; ~3 weeks of work left; 3-person team, one UR3 rig)
PHANTOM: a tactile **world-action model** (WAM) teacher and a **sensor-free student** distilled from it.
Claim we want to make: anticipating contact from vision+proprio (learned from a tactile teacher) closes the loop
on grasping better than vision-only imitation.

## Architecture (as built, repo ~/GitHub/phantom, all real)
- Backbone: NVIDIA **Cosmos-Predict2.5-2B** action-conditioned video DiT, LoRA r16 (22.9M trainable) + 1.5M new params.
  Sequence layout (T=14 latent frames, 4480 tokens): video_cond [0:1) | video_gen [1:4) | obs_gel | obs_mech |
  obs_proprio | contact [7:10) | action [10:14). It predicts FUTURE VIDEO latents and the ACTION CHUNK jointly.
- Actions: **flow matching (rectified flow)** over a chunk of **16 steps at 10 Hz** (1.6 s), action = Δ-EE (6) + gripper (1),
  sampled with **NFE=5**, guidance 1.0 (classifier-free obs guidance), persistent noise across replans. No value function.
- Tactile: two **DM-Tac visuotactile gel sensors** (Daimon; fields = displacement/force maps) on a Robotiq 2F-85;
  **HHT** tactile encoder; **ACC** = anticipatory contact classifier (gate p(contact soon) + event classes) with a
  reactive term; **ACE** heads predict a "contact package" (cpk: contact events, CoP, wrench) for the next chunk and the
  previous chunk's package is fed back at the next replan (prev_cpk). Losses: action_v_mse (flow), video_v_mse,
  contact heads (event CE, gate BCE, wrench/CoP), an optional event-band loss. Proprio input `ur_state` =
  [q(6), qd(6), tcp_pose(6), tcp_speed(6), gripper pos+OBJ] normalised by dataset stats; wrist F/T window.
- Student (planned, zero experiments so far): same network without tactile inputs, distilled from the teacher
  (trajectory/event/behaviour distillation losses, "HID" = hidden-feature alignment ablation). Baselines
  (vision_only, no_distill) are NOT trained yet.
- Training: teacher v4 = 20k steps on 790 demos (from scratch, LoRA), v5 = 3k-step fine-tune of v4 on 1115 demos
  (790 + 325 new: 280 ordinary successes + 45 under-grasp failure demos with action loss weight 0 but contact heads
  supervised). 1×H100, batch 4×2, ~7 s/step. Offline terminal metric improved 20.7→17.4 mm endpoint error.
- There is DAgger scaffolding (phantom/dagger: rollout/relabel/manifest) and an AWR-style weighting config
  (beta_awr, max_weight) that has never been exercised on the rig.

## Data
- 4 tasks on a UR3 + Robotiq 2F-85 + RealSense scene camera (near top-down oblique view): pick **waffles** pack,
  **Carton**, **egg**, **whiteboard** eraser (grasp then place/wipe), ~250 successful teleop demos each (Echo teleop),
  demo length 16–31 s, grasp at t≈5–8 s, close height z≈72 mm (waffles). Start poses tightly repeated (joint std 0.5–7°).
- Deploy episodes are recorded with the same schema (zarr streams + planner_trace.json with every sampled chunk).

## Rig results (the problem)
- Session 08-20 (v4): 13/13 grasps closed 35–80 mm above the object. Session 08-28 (v4 and v5, 26 eps): 0 successes.
- Decomposition (08-28): the **executor tracks the commanded chunks within mm** — the policy itself decelerates and
  stops 65–120 mm above the object, closes on air to a demo-like aperture, then LIFTS and transports to the bin
  while its own tactile contact gate correctly says "nothing there" (p_none 0.99). Then it retries. Offline on
  held-out demo states the same model commits correctly (17 mm endpoint error) ⇒ pure covariate shift /
  compounding error of behaviour cloning + the action head never conditions on "did the close succeed"
  (all training closes succeed; failure demos carry no action signal).
- Rollouts descend ~30% slower than demos (33 vs 45–50 mm/s); one noise seed descended at full speed to the right
  height ⇒ behaviour is multi-modal across seeds. Suspects: velocity self-consistency via proprio (slow arm ⇒ slow
  mode), z-ambiguity of the near-top-down camera, weak proprio reliance.
- 16/26 episodes today were invalid (arm in a wrapped-wrist / flipped-IK joint configuration = raw-joint OOD); fixed by
  a joint-space start gate + z floor + STOP hitbox. Inference stack: 0.9 s replan latency, RTDE servoL at 125 Hz,
  chunk blending, speed governor from predictive sigma.
- Budget: rented H100s (~$2–3/h), ~34 h for a from-scratch 20k-step teacher; rig time is the scarce resource
  (one rig, 2 operators, ~15–20 rig-hours total available). Real-robot RL with 300–1000 trials is NOT feasible;
  50–150 rollouts with automatic tactile success labels is.

## What we need from the literature (for each topic)
Concrete, sourced answers to: what has worked on real robots for exactly this failure mode; what the
sample/time budgets were; what is the minimal change to OUR stack (flow-matching chunk policy + tactile heads +
video world model) that captures it; what would make reviewers see novelty. Do not propose infrastructure
rewrites. Prefer 2024–2026 papers with real-robot results; cite arXiv ids.
