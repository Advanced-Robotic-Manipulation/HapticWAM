# Lens: DEPLOY RECIPE END TO END — PHANTOM @ d9c40f2

Everything below was executed on the Mac with `.venv/bin/python`, mock drivers
(`configs/hardware.yaml` mode: mock), `--tiny`, `configs/paths.local.yaml`.
Scratch: `/private/tmp/.../scratchpad/validate/scratch/deploy-recipe/`.

## 1. What works (verified by running it)

### 1.1 The Session-4 arms both run headlessly on the mock rig

    .venv/bin/python -m phantom.scripts.run_deploy --system teacher --tiny --task waffles \
        --episodes 2 --device cpu --no-label-prompt --max-replans 3 --out .../runA --seed 4242
    .venv/bin/python -m phantom.scripts.run_deploy ... --nfe 1 --terminal-veto            # arm B
    .venv/bin/python -m phantom.scripts.run_deploy ... --nfe 1 --terminal-veto \
        --parity-fixes --k-seeds 2 --no-hitbox --max-tcp-speed 0.15                        # arm C

All three complete; arm C ran 6 replans with `parity:on`, `kseeds:2` and no
exceptions. `pytest tests/test_deploy_levers.py tests/test_deploy_parity_fixes.py
tests/test_rig_leftovers.py tests/test_rig_safety_0828.py tests/test_safety.py
tests/test_small_fixes_0829.py tests/test_deploy_fixes.py -q` → **151 passed**.

### 1.2 Meta condition tags carry the whole envelope

runA ep0 / ep1 `meta.json` tags:

    ['nfe5','g1.0','freshnoise','ckpt:','git:d9c40f2','zfloor:42mm','hitbox:30mm',
     'vmax:0.25','parity:off','veto:off','kseeds:1','seed:4242','unlabeled']
    [... 'seed:4243' ...]

runB (unseeded, two episodes): `seed:2098336675`, `seed:3100725310` — the
per-episode fresh draw works and differs. runC: `vmax:0.15`, `hitbox:none`,
`parity:on`, `kseeds:2`, `veto:pc0.50/pn0.90/r3`. `meta.deploy_overrides` carries
the hitbox box and the z floor; the episode is stamped with the BASE hardware
hash. Every item the lens asked for (seed/zfloor/hitbox/vmax/parity/veto/kseeds/
nfe/ckpt) is present.

### 1.3 The sampler really is reseeded, and `--seed` really is reproducible

`scratch/deploy-recipe/seed_probe.py` builds the tiny policy through
`run_deploy.build_policy`, then replays exactly what the episode loop does
(`rf._gen = torch.Generator().manual_seed(ep_seed); rf.reset_episode_noise()`):

    same seed, two episodes -> max|diff| = 0.0
    seed 4242 vs 4243        -> max|diff| = 1.7963457
    replan0 vs replan1 within an episode (persistent-noise) max|diff| = 0.0084
    unseeded episode seeds: [96994494, 3233721230, 4005080585, 2543396944] unique: 4
    seeded  episode seeds: [4242, 4243, 4244, 4245]

The generator is a CPU `torch.Generator`, and every draw in `rf.sample` /
`build_x0` / `_action_strip_noise` goes through `self._gen` then `.to(dev)`, so
the seed is device-portable (5090 == NUC == Mac).

### 1.4 `EXTRA` appending: the later `--nfe` wins

    parse_args('... --nfe 5 --guidance 1.0 --nfe 1 --terminal-veto')
      -> nfe 1 veto True pnoise True ema True seed None
    parse_args('... --nfe 5 --guidance 1.0 --seed 4242')
      -> armA nfe 5 seed 4242 veto False

argparse last-wins holds; the doc's claim is correct.

### 1.5 Safety layer

* **z floor is a CLAMP, not a STOP** — the §1.11 fix landed.
  `--z-floor 0.49` on the mock → `SAFETY workspace_clamp value=0.522 -> clamp`,
  five replans, `stop=None`. `safety.py:207` evaluates the hitbox on
  `self.clamp_target(...)`, and `apply_hitbox` runs *before* `apply_z_floor`
  so the hitbox floor (0.022 m) sits below the clamp floor (0.042 m).
  `WorkspaceBox.contains` is inclusive, so a target clamped exactly onto a
  shared boundary does not trip the hitbox.
* **hitbox STOP works** — the mock arm starts at z = 0.50 m, above the waffles
  demo envelope ceiling 0.4401 m → `SAFETY hitbox_exit value=0.506 -> stop`,
  episode ends `safety_stop`.
* **speed cap** — `--max-tcp-speed 0.15` reaches `hw.arm.limits.tcp_speed_m_s`
  (executor.py:275) and is tagged `vmax:0.15`.
* **joint gate** — `start_sigma_report` with `q[5] += 2π` returns worst
  sigma **72.0** and prints the `!!! q6 is +360 deg from the demos — a FULL-TURN
  wrap` line. `configs/start_poses.yaml` has `n=250` per task (regenerated).
  Note it is only reached under `arm_real`, so the mock run does not exercise it.
* **`--parity-fixes` ring depth** — `_n_arm` grows to
  `ceil(1.2·1.7·rtde_hz)+8 = 1028` rows; the ring holds `ring_seconds·rate`
  = 2500 rows (5 s) / 10000 (nuc, 20 s). No truncation.

### 1.6 planner_trace keys

runB/runC rows carry `t, latency_s, gate, p_evt, sigma, accepted, actions,
terminal_veto, diag`. `terminal_veto` = `{'p_contact':…, 'retries':…,
'action':…, 'at_floor':…}`; `diag` = `{'nfe':1,'guidance':1.0,'k_seeds':2,
'head_dz_spread_mm':2114.61,'k_rejected':0,'k_pick':0}`.

## 2. What is still wrong

### 2.1 [HIGH] `--terminal-veto` is a no-op on the recorded rig failure

`planner.py:403`:

    closing = g_max > v.close_pos and (g_max - grip_now) > v.close_rise

`grip_now` is the *measured aperture right now*, so the rise is measured per
replan. The recorded rig terminal phase ramps the aperture **0.31 → 0.52 over
several replans while still descending** (docs/rig_session_v5.md; SYNTHESIS §2).
`scratch/deploy-recipe/veto_ramp.py` replays that ramp:

    replan 0: z=190mm grip_now=0.31 chunk_gmax=0.35 -> {'action': 'none'}
    replan 1: z=164mm grip_now=0.35 chunk_gmax=0.39 -> {'action': 'none'}
    replan 2: z=138mm grip_now=0.39 chunk_gmax=0.44 -> {'action': 'none'}
    replan 3: z=112mm grip_now=0.44 chunk_gmax=0.48 -> {'action': 'none'}
    replan 4: z= 86mm grip_now=0.48 chunk_gmax=0.52 -> {'action': 'none'}
    closed_at armed? None  retries 0

p_evt[none] = 0.99 throughout. The close mask never fires, `closed_at` is never
armed, and therefore the phantom-grasp recovery **can never run**. Arm B would
be recorded with `veto:pc0.50/pn0.90/r3` in `meta.json` and `"action":"none"` in
every trace row while closing on air exactly as v5_6 did on 08-28 — an ablation
that reads "veto on" and measures nothing.

Training's own close detector (`train/common.py:358`) compares against a
**running minimum** (`pos - run_min > CLOSE_ABS_RISE`), not against the current
sample. Minimal fix: carry `g_min` in `veto_state`, update it each replan, and
test `(g_max - state["g_min"]) > v.close_rise`.

### 2.2 [HIGH] the veto's recovery arm never expires → it reopens a *successful* grasp

`planner.py:379`: `if state["closed_at"] is not None and p_none > v.p_none`.
`closed_at` is set on an accepted `close_allowed` and cleared **only** inside the
recovery branch — nothing compares it against the current replan time, despite
the docstring's "the very next replan". `scratch/deploy-recipe/veto_probe.py`:

    r0 {'action': 'close_allowed'}          # p_none 0.05
    r1..r3 {'action': 'none'}  closed_at 0.0   # transport, contact holds
    r4 (transient p_none=0.95, 4 replans AFTER the close): {'action':'recovery_open'}
       gripper channel now: [0.23 0.23 0.23]   z deltas now: [0. 0. 0.]

On a genuine grasp, any later replan whose gate reports `p_evt[none] > 0.9` —
release, a transport frame where the gel loses the object, an egg held lightly —
commands the gripper open to the task start aperture and zeroes the lift. Three
of those and the episode ends `veto_retry_cap`. That is a dropped/damaged object
and a success recorded as a failure, biasing the A/B against the arm being
promoted. Fix: consume the arm once (`closed_at = None` at the end of the
following replan) or gate on `plan.t_created - state["closed_at"] <= 1.5·period`.

### 2.3 [HIGH] the documented Arm A re-creates the constant-seed bug

`docs/rig_session_v5.md:71` — `EXTRA="--seed 4242" ./GO_v5_waffles.sh 1`, i.e.
**one episode per process**, repeated ≥10 times per the interleaving protocol.
`run_deploy.py:319`, `episode_seed(base, i) = base + i` where `i` is the
*within-process* index, so every Arm-A launch gets `i = 0` → `seed 4242`. With
the GO script's `--persistent-noise` that is **one identical noise tensor for
every Arm-A episode of the session** — the exact failure `ba61354` exists to
remove. Evidence: `episode_seed(4242,0) == 4242`; runA ep0 tag `seed:4242`, and
seed_probe shows two episodes at the same seed are bit-identical (max|diff| 0.0).
Fix: drop `--seed` from Arm A (the unseeded path already records the draw), or
run the arm's whole block in ONE process (`./GO_v5_waffles.sh 10`), or fold a
per-process nonce into `episode_seed`.

### 2.4 [MEDIUM] `phantom/eval/trial_runner.py` never reseeds

`_run_one` (`trial_runner.py:90`) calls `rt.run_episode` and never touches
`policy.rf._gen`; the campaign's `seed` column is a ledger label only (the
§1.11 finding, fixed in `run_deploy` and *not* here). Every trial of a campaign
runs off rf.py's constructor `manual_seed(0)` stream, no `seed:` tag lands in
`meta.json`, and a resumed campaign (the runner is explicitly resumable) restarts
the generator at 0 and replays the same noise draws. This is the path D15/D16's
main matrix and `aggregate.py`'s seed-clustered bootstrap read.

### 2.5 [MEDIUM] the veto's "already low enough to close" band is the safety floor, not the demo close band

`planner.py:320` `z_margin = 0.015` (hard-coded; `build_veto` never passes it, so
there is no CLI knob) and `:407` `at_floor = z_now <= v.z_floor + v.z_margin`.
`z_floor = tcp_z_min − 10 mm`, so the band tops at **57 mm (waffles), 81 mm
(Carton), 80 mm (whiteboard), 65 mm (egg)** while `grasp_label.Z_MAX_MM` — the
demo p95 close height + 15 mm — is **103 / 144 / 181 / 101 mm**. A legitimate
demo-height close with an under-confident gate is masked:

    waffles close at z=65mm (demo band 46-88mm), p_contact=0.40 -> 'close_masked'

Separately, docs/rig_session_v5.md:69 describes the escape as "the gripper is at
the demo grasp band" — an *aperture* rule the code does not implement.

### 2.6 [MEDIUM] the per-seed head descents are filtered out of planner_trace.json

`planner.py:496` keeps only `isinstance(v,(int,float,str,bool))` diag entries, so
`_select_seed`'s `head_dz_mm` list (the only record of what the K seeds actually
proposed) is dropped; only `head_dz_spread_mm`/`k_rejected`/`k_pick` survive.
Verified on runC's trace and reproduced directly. That is the quantity
`replay_rig`'s K-seed spread has to be validated against (gate G0) and the only
way to attribute the K-seed lever after the session.

### 2.7 [MEDIUM] `--seed` does not make a trial reproducible: the start pose jitter is a separate, unseeded RNG

`run_deploy.py:553` `rng = np.random.default_rng()` (no seed) → `move_to_start` →
`sample_start_pose` jitters the homing target by up to ±1σ of the demo start
distribution: for waffles that is **±26.9 / ±29.0 / ±27.1 mm** in x/y/z plus
±1σ on the aperture. The A/B's "paired per placement" assumption and the primary
number (miss distance, z-at-close) are both of that magnitude. The realised pose
is recoverable from the recorded arm stream, but it is not seeded, not tagged,
and `--seed`'s help ("reproducible noise draws for paired trials") reads as if it
were.

### 2.8 [LOW] K-seed selection scores contact from seed 0

`policy.py:256-258`: `p_evt0 = pred.acc.p_evt[0]` — row 0 of the K-expanded
batch — decides whether the "reject the timid mode" branch runs at all, while
each of the K rows carries its own gate output and the selected row is `j`.

## 3. Verdict

**NOT READY.** The safety envelope, the per-episode seeding, the tagging and the
`--nfe`/`--parity-fixes`/`--k-seeds` plumbing all work end to end on the mock rig,
but the recipe's two headline levers are broken in opposite directions: the
terminal veto never arms on the aperture ramp the rig actually produces (2.1) and
fires spuriously long after a real close (2.2), and the documented Arm A command
(`--seed 4242` with one episode per process) reproduces the constant-seed bug the
session exists to escape (2.3).
