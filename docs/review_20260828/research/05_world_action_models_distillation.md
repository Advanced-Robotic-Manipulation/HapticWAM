# World-Action Models & Privileged-Sensor Distillation — Lit Review

Scope: is PHANTOM (Cosmos-Predict2.5-2B LoRA, joint video+action+contact prediction, flow-matching
action chunks, ~1.1k demos) a legitimate "world-action model" (WAM) in the 2025-2026 sense, does the
joint video head plausibly help or hurt the closed-loop grasp-commit failure we're seeing, and what does
the distillation literature say about a tactile-teacher -> vision-only-student pipeline.

## 1. Papers table

| Title (arXiv id, year) | Method | Data scale | Real-robot numbers |
|---|---|---|---|
| **Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning** (2601.16163, 2026) | Post-trains Cosmos-Predict2 (the exact backbone family we use) with **zero architectural changes**: actions, proprio, and a value scalar are encoded as additional *latent frames* interleaved with image frames in one diffusion sequence — sequence roughly `[null, cur-proprio, cur-images(x3), action-chunk, future-proprio, future-images(x3), value]`. Inference uses **5 denoising steps for actions, 1 step for future state, 1 step for value** (i.e. video/state heads are cheap single-step, action head gets more NFE — the inverse of what you'd guess). | RoboCasa 50 demos/task ("6x fewer demos than baselines"), LIBERO ~2000 demos total, **real ALOHA bimanual: ~185 demos across 4 tasks**, 48h on 8×H100. | LIBERO 98.5%, RoboCasa 67.1% avg; "highest average score" on real bimanual tasks (no per-task % surfaced in what we could pull, cookbook page doesn't report it plainly). |
| **HarmoWAM: Harmonizing Generalizable and Precise Manipulation via Adaptive World Action Models** (2605.10942, 2026) | Names our exact tension: "**Imagine-then-Execute**" (predict video, then infer action from it) generalizes well for *transit/approach* but is imprecise at *contact* (~60% success); "**Joint Modeling**" (predict video+action together, like us) is precise (80%+) but fails to explore OOD approach configurations (<60% transit). Fix: two action experts (Reactive: acts directly off predicted frames/features, for open-loop exploration; Predictive: uses latent video dynamics for fine-grained contact) behind a **Process-Adaptive Gate** that switches expert by task phase (transit vs. interaction). | Pretrain ~1.9M trajectories (DROID+AgiBot+RoboMIND+proprietary) then robot-specific finetune. | In-domain 89% vs Cosmos-Policy 78%; zero-shot OOD 82% (33% over prior VLAs); position-OOD 80% vs 26% baseline. Ablation: removing world-model latent features drops Predictive expert ID 95%→62%. |
| **ViPRA: Video Prediction for Robot Actions** (2511.07732, 2025) | Video-language model jointly predicts future visual tokens + motion-centric latent actions; **crucially, this joint prediction happens only at pretraining**, then a separate chunked flow-matching decoder maps latent actions → continuous actions at finetune/deploy time (decoupled, like UVA). Explicitly tested keeping the coupling at finetune time too. | Finetune on 100-200 teleop demos. | SIMPLER +16%, real-world manipulation +13%. Ablation (Table 2): removing state/video prediction at pretrain: 69.8%→59.4% (AR) / 62.5%→53.2% (FM). **Adding state prediction back in at finetune time collapses performance to 53.1%/31.3%** — the authors attribute this to "the autoregressive structure coupl[ing] action prediction with video prediction, causing compounding errors that drift irrecoverably on out-of-distribution states." |
| **Unified Video Action Model (UVA)** (2503.00200, 2025, Stanford) | Learns one joint video-action *latent*, then **decouples decoding** into two lightweight diffusion heads so action inference at test time never has to run video generation (bypasses the video decoder for speed). Masking lets one model do policy / forward dynamics / inverse dynamics / video-gen. | Large multitask robot datasets + real-world; exact demo counts not in the fetched excerpt. | Reports matching or beating action-only baselines "without compromising performance," faster inference than joint-decode approaches (numbers not captured in fetch — recommend full-PDF read). |
| **Unified World Models (UWM)** (2504.02792, 2025, RSS) | Single transformer, **independent diffusion timesteps for video vs. action** — by choosing which modality's timestep is "denoised" you get a policy, forward dynamics, inverse dynamics, or a video generator from the same weights. Trains on both action-labeled robot data and action-free video. | Large-scale multitask robot data + video. | Real-world: best of 5 tasks, up to +20% over best baseline; notably robust to OOD visual distractions. |
| **WorldVLA: Towards Autoregressive Action World Model** (2506.21539, 2025) | Autoregressive tokenized world model + action model in one framework; each helps the other (world model conditions on actions to predict frames, action model conditions on images). **Documents a WorldVLA-specific failure mode**: autoregressive multi-action-chunk generation degrades because "limited generalization capability for action prediction" propagates errors chunk-to-chunk; fixed with an **attention mask that hides prior generated actions** when generating the current action. | Not captured in fetch. | "Outperforms standalone action and world models" — exact numbers need full PDF. |
| **Video Prediction Policy (VPP)** (2412.14803, 2024→ICML'25) | Doesn't generate pixels for the robot — reuses the **internal features of a video diffusion model (Stable Video Diffusion)** as the policy's visual representation, arguing VDM features already encode predicted dynamics, then trains an MDT-style diffusion action head on top. Finetunes on robot + internet human-manipulation video. | CALVIN ABC-D + real dexterous tasks. | CALVIN: +18.6% relative over prior SOTA; real dexterous tasks: +31.6% success. Ablation swapping the VDM encoder for VC-1/Voltron/plain VAE features shows VDM features alone drive most of the gain (4.33 vs 1.2-2.6 CALVIN avg length) — i.e. the *prediction-trained representation*, not explicit rollout, is what helps. |
| **GR-2: A Generative Video-Language-Action Model with Web-Scale Knowledge** (2410.06158, 2024) | Pretrain on 38M internet video clips (video generation objective only) → finetune jointly for video generation + action prediction on robot trajectories. | 100+ tasks, large in-house robot dataset. | 97.7% avg success across 100+ tasks; strong generalization to novel backgrounds/objects (numbers for the internet-pretraining ablation not captured here). |
| **GR-1** (2312.13139, 2023 — GR-2's predecessor, worth citing as the origin of this recipe) | Same recipe one generation earlier: video-generative pretrain → finetune to jointly predict actions + future frames. | CALVIN. | CALVIN 88.9%→94.9%; zero-shot scene generalization 53.3%→85.4%. |
| **DreamGen** (2505.12705, 2025, NVIDIA) | Doesn't put video-gen in the policy at all — uses an image-to-video model to hallucinate whole new demonstrations ("neural trajectories") in novel scenes/behaviors, then extracts pseudo-actions via a latent-action or inverse-dynamics model, and trains an ordinary action-only policy on the synthetic+real mix. | **1 real pick-place task, 1 environment**, teleop-only. | Humanoid performs 22 new behaviors in seen+unseen envs from that single seed task. Introduces DreamGen Bench showing video-gen fidelity correlates with downstream policy success — i.e. this is the paper to cite if you want to argue "video quality predicts action-policy quality," relevant to whether OUR video head is even generating anything useful given LoRA-r16 + ~1k demos. |
| **Genie Envisioner** (2508.05635, 2025, AgiBot) | Full platform, not just a policy: GE-Base (instruction-conditioned video diffusion world model) → GE-Act (lightweight **flow-matching decoder**, structurally close to what we do) mapping GE-Base's latent to actions → GE-Sim (the same video model reused as a closed-loop simulator for evaluation). | Large-scale AgiBot data. | Not captured in fetch (recommend reading directly if closed-loop sim-eval matters for your rig-time bottleneck — GE-Sim is literally "evaluate policies without rig time"). |
| **PTLD: Sim-to-Real Privileged Tactile Latent Distillation** (2603.04531, 2026) | Teacher: RL (Asymmetric Actor-Critic) policy in sim with privileged object pose/shape, tactile *bypassed entirely* in sim. Student: real tactile+proprio encoder trained online (inside the AAC loop, not offline BC) to match teacher's latent via `L_latent = ||E(tactile,proprio) - sg(E(privileged))||`, i.e. **hidden-feature/latent alignment**, combined with PPO. | ~180 min real trajectories, 30 traj/iter × 3 DAgger iters, Allegro hand + Xela tactile skins. | In-hand rotation: 4.5±2.3 vs 0.5±0.5 (proprio-only) reward; reorientation goal completion 3.3±1.55 vs 2.1±2.14. Ablation: the **online** latent-alignment loss (trained jointly with the RL objective) beats offline two-stage distillation — the loss has to stay live during policy training, not just be a post-hoc BC target. |
| **HapticVLA: Contact-Rich Manipulation via VLA without Inference-Time Tactile Sensing** (2603.15257, 2026) | **Directly your architecture's shape**: two stages — (1) safety-aware reward-weighted flow matching using the tactile-equipped teacher, (2) tactile distillation that compresses the teacher's tactile signal into a compact **tactile token**, and trains the student VLA to *predict that token from vision+state alone* (a supervised auxiliary target, not just BC on actions). ⚠ Given the title/timing this may overlap with or be your own team's HapticVLA (IROS 2026) paper — worth checking for direct method reuse/differentiation before citing as "prior work." | Not captured in fetch — recommend full read given the near-identical framing to PHANTOM's planned student. | — |

## 2. Is "Cosmos-Predict2.5 + LoRA + joint video/action/contact head on ~1k demos" a WAM reviewers will accept?

**Yes on architecture, with one gap.** The 2025-2026 WAM literature has converged on a small number of
recognizable patterns, and PHANTOM matches the dominant one cleanly: encode future observation(s), action
chunk, and auxiliary signals (value/contact/proprio) as co-located tokens in one video-DiT sequence and
denoise them jointly or near-jointly. This is *exactly* Cosmos Policy's recipe (2601.16163) — actions,
proprio, and a scalar (value there, contact-package here) as latent frames alongside video frames, same
backbone family. Reviewers who know this literature (very likely for ICRA 2027) will recognize the pattern
immediately; the paper should cite Cosmos Policy explicitly as the closest architectural relative and
differentiate on the axis that actually matters for your claim: **you replace their scalar "value" frame
with a structured tactile "contact package" (events, CoP, wrench) and use it for teacher→student
distillation, not RL value estimation.** That's a legitimate, citable delta, not just backbone reuse.

The gap: none of Cosmos Policy, GR-1/GR-2, UWM, or WorldVLA report training at your scale (~1k demos,
LoRA r16, 2B backbone). ViPRA is the closest data-scale precedent (100-200 demos) and is the single most
important comparison paper for you, because it ran the ablation you need and haven't run:
**joint video/action coupling helps at *pretrain* time but actively hurts when the coupling persists into
finetune/deployment**, with the collapse explained as compounding-error drift on OOD states (53%→31% in
their AR/FM conditions respectively). That description — a policy that behaves fine in-distribution and
drifts irrecoverably once slightly OOD — is a near word-for-word match for PHANTOM's rig failure (offline
17mm endpoint error, closed-loop 0/26 with the executor tracking perfectly and the *policy itself*
stopping short). HarmoWAM independently frames the same tension as "Imagine-then-Execute" (generalizes,
imprecise) vs "Joint Modeling" (precise, fails to leave the training distribution) and reports that pure
joint modeling gets stuck exactly the way you're describing — good ID precision, poor behavior once the
rollout state distribution shifts even a little.

**Cheapest ablation to settle this**: freeze the diffusion action head's conditioning to bypass the
video_gen tokens at inference (i.e., set NFE=1 or zero out video_gen conditioning / replace with the
teacher-forced last true frame) and compare grasp-commit distance on the *same replan states* recorded in
your existing planner_trace.json rollouts. If commit height doesn't change, the video head isn't the
locus of the failure (points instead at the flow-matching commit signal / lack of failure-conditioned
action targets, per your own decomposition). If commit height improves, ViPRA's diagnosis directly
applies: the persistent coupling is degrading action decisiveness on OOD (near-contact, low-tactile-
evidence) states, and the fix is architectural (decouple video-gen and action decoding at inference like
UVA/ViPRA do), not a data problem. This needs zero new data collection — it's an inference-time ablation
against existing logs plus one offline eval pass, which fits your token/rig-time budget far better than
new rollouts.

## 3. What the strongest 2025-2026 WAM papers do that PHANTOM doesn't

- **Decouple video-gen from action decoding at inference** (UVA, ViPRA). PHANTOM currently keeps them
  coupled through every replan. This is the single highest-leverage, ~1-week change: add a second,
  action-only flow-matching head that shares the trunk but doesn't require sampling video_gen tokens at
  deploy time, and A/B it against the current joint head on existing eval states. Cosmos Policy's own
  NFE schedule (video/value get 1 step, only actions get 5) is a soft version of this — it already treats
  action as the expensive modality and world dynamics as cheap/approximate, which suggests going further
  toward "video head only guides representation, doesn't get resampled by RTDE frequency" is compatible
  with the family, not a departure from it.
- **Phase-adaptive gating between "explore/approach" and "commit/interact" behavior** (HarmoWAM). This is
  the most directly transferable idea to your failure mode: you already compute an anticipatory contact
  gate (ACC, p(contact soon)) but it isn't currently used to *gate the action-generation regime* — it's a
  classifier output, not a control-flow switch. Using ACC's own gate value to switch between a
  "descend/approach" execution mode and a "commit/close" execution mode (even just switching guidance
  scale or persistence-of-noise seed near p(contact)~1) is a ~1-week change that reuses infrastructure you
  already have and maps directly onto a validated idea in the literature. This is arguably a *better*
  fit for your paper's contribution story than pure distillation, because it uses the tactile teacher's
  own contact signal as the gate — nobody else gates on a *tactile* anticipation signal specifically.
- **Failure-conditioned action targets during training** (WorldVLA's masking fix is adjacent, not
  identical — it fixes compounding error across an autoregressive *action* sequence, not a BC data
  problem). Your own diagnosis — "all training closes succeed; failure demos carry no action signal" —
  is not directly solved by anything in this table, but DreamGen's mechanism (synthesize additional
  trajectory variation cheaply) suggests a cheap fix: since you already have real rig failure logs
  (planner_trace.json from the 26-episode session), those are *real* off-policy states with known bad
  outcomes — hindsight-relabel them with a "commit harder" auxiliary loss (DAgger-lite, which you already
  have scaffolding for) rather than trying to source more successful demos. This doesn't require new
  video-gen synthesis at all, just closes the loop on data you've already collected.
- **Closed-loop sim-as-simulator evaluation** (Genie Envisioner's GE-Sim: reuse the video world model
  itself as a rollout simulator to screen policy checkpoints before spending rig time). Given your stated
  bottleneck ("rig time is the scarce resource... 15-20 rig-hours total"), this is worth a half-day
  feasibility check even if not adopted: does replanning against your own model's *predicted* video_gen
  frames (instead of the real camera) produce a rollout whose failure mode matches the real one? If yes,
  you get free offline A/B between the "decoupled head" fix and the "ACC-gated" fix before burning rig
  hours on either.

## 4. Distillation: what has real-robot evidence, and a minimal student experiment

Evidence quality, ranked:
- **PTLD (2603.04531)**: real Allegro-hand results, but the transferable lesson is about *training
  regime*, not loss form — the latent-alignment loss (`||E_student − sg(E_teacher)||`) only worked well
  when kept **online** inside the same RL loop as the student's own objective; a two-stage
  "distill-then-freeze" approach underperformed. Your planned distillation is offline (teacher already
  trained, student trained after) — PTLD's finding is a caution, not a green light: budget for the
  possibility that offline BC-style distillation of the tactile latent underperforms an online/joint
  variant, and have a fallback (e.g., distill early and keep a small joint fine-tune phase rather than
  fully freezing the teacher-derived target).
- **HapticVLA (2603.15257)**: closest to your literal plan — compress tactile signal to a token, train
  student to predict the token from vision+state as an **auxiliary supervised loss alongside action loss**
  (not just matching a hidden feature — matching a semantically compact target). This is more promising
  than raw hidden-feature MSE (HID ablation) because a discrete/compact "contact package" target (which
  you already compute — cpk: events, CoP, wrench) is more interpretable and easier to weight against the
  flow-matching action loss than an unconstrained hidden vector. **Recommend prioritizing "predict cpk
  from vision+proprio" as the primary distillation loss over raw hidden-feature (HID) alignment.**
- Other dexterous-hand privileged-distillation papers found in search (RobustDexGrasp 2504.05287,
  DexGrasp-Zero 2603.16806) follow the same recipe (sim teacher with full contact truth → real-observable
  student via BC) but are sim-teacher, not real-tactile-teacher like yours, so they're weaker precedent
  for the "does the *specific modality gap* (real tactile → vision-only) transfer" question than PTLD or
  HapticVLA.

**Minimal credible student experiment given ~3 weeks and one rig**: don't train a from-scratch
vision-only baseline first (expensive, and you don't have baselines trained yet per your own status).
Instead: (1) freeze the v5 teacher, (2) initialize the student from the *same* LoRA weights minus tactile
input tokens (obs_gel/obs_mech removed from the sequence), (3) train it with two losses — action_v_mse
(standard) + a new `cpk_pred` loss (student predicts the teacher's contact-package output from vision+
proprio only, i.e. HapticVLA-style token prediction, not raw HID feature matching) — for as few steps as
your v5 fine-tune took (~3k steps was enough to move 20.7→17.4mm), then run the SAME cheap inference-time
ablation from §2 (video-head bypass) on the student before spending rig hours. If the student's offline
endpoint error is within noise of the teacher's, you have a paper-ready result without new rig time; if
it's much worse, that's itself informative (contact package is doing real work the model can't recover
vision-only) and worth reporting even negative.

## 5. Risks to the paper's claims

- **The core empirical claim ("tactile anticipation closes the loop better than vision-only") is currently
  unfalsifiable** — no vision-only baseline exists yet, and the teacher itself fails closed-loop at 0/26.
  A reviewer's first question will be "your teacher doesn't work on the rig — how do you know the tactile
  signal is the reason a student would do better or worse?" You need the teacher to succeed at some
  nonzero closed-loop rate before a student comparison means anything; §2's cheap ablation and §3's
  ACC-gating idea are the fastest paths to a working teacher baseline within budget.
- **ViPRA and HarmoWAM both suggest your exact symptom (fine offline, drifts/hesitates OOD when
  video+action are coupled at deployment) is a known, structural failure mode of the "joint modeling"
  family your architecture belongs to** — meaning if you don't change the coupling, a reviewer familiar
  with either paper can predict your result from your architecture description alone, undermining novelty
  of "we found a failure mode" (you'd need to frame it as "we confirm and additionally fix it via X").
- **Small-N statistics**: 13/13 vs 0/26 across two sessions with different hardware states
  (16/26 invalid due to IK config that session) makes the "v4 worked, v5 didn't" comparison confounded by
  the joint-config bug fix happening between sessions, not just model/data changes — flag this explicitly
  rather than let a reviewer find it.
- **HapticVLA name/timing collision**: if 2603.15257 is not your own paper, its framing (tactile token
  distilled into vision+state student) is close enough to your planned contribution that you should read
  it in full now and stake out an explicit delta (e.g., closed-loop world-action video model vs their
  presumably BC-only VLA; structured cpk with CoP/wrench vs a single compact token) before a reviewer
  flags it as prior art.

## 6. Top 3 papers to read in full

1. **ViPRA (2511.07732)** — the ablation that most directly predicts and explains your rig failure;
   read Table 2 and the discussion of why finetune-time coupling collapses performance.
2. **HarmoWAM (2605.10942)** — names your exact architecture family's known weakness and proposes a gate
   mechanism you can approximate cheaply with your existing ACC contact gate.
3. **Cosmos Policy (2601.16163)** — your literal backbone's own policy paper; read the full method for
   the token-layout and NFE-schedule details this fetch couldn't fully extract (particularly whether they
   report any per-modality ablation), since it's the paper reviewers will most directly compare you to.
