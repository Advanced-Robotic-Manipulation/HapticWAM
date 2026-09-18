# tools/rig/box — the scripts that only existed on the rig box

Operator scripts that lived **only** in `~/phantom-icra-2027/` on compute3 (the 5090 box that
drives the UR3) and were never tracked. `../README.md` covers the launcher set that the box
reaches through symlinks (`PICK.sh`, `GO_ANY.sh`, `MODELS.tsv`, `stage_v5.sh`, `SNAP_ANY.sh`,
`GRIPPER_*.sh`, `screenshot_*.sh`, `probe_rig.py`); this directory is the rest — the scene
check, the menu maintenance, the pi0.5 baseline install, the hub mirroring and the offline
replay batteries. Session recipe and safety rules stay in `docs/rig_session_v5.md`.

These are **tracked copies, not the live ones**: the box copies are what actually runs. Nothing
here is wired into the symlinks, so changing a file here does not change the box — copy it over
deliberately (and vice versa).

## How the box is laid out

```
~/phantom-icra-2027/
  phantom/                  this repo (the clone the rig runs from; .venv lives here)
  PICK.sh  EXPERIMENT.sh  SERVE.sh  serve_bg.sh  GO_ANY.sh  READ_TCP.sh
                            -> symlinks into phantom/tools/rig/
  MODELS.tsv                the LIVE policy menu the launchers read (see below)
  refs/ref_<task>.png       training reference frames, for the scene check
  data/episodes/deploy/     output: <YYYYMMDD>/ for ordinary days,
                            <YYYYMMDD>_experiment/ for EXPERIMENT.sh blocks
  logs/  runs/              server + setup logs; box-local checkpoints
  pi05venv/  baselines_venv/  separate venvs for the LeRobot baselines
```

`MODELS.tsv` is `label<TAB>ckpt<TAB>note<TAB>system<TAB>extra-flags`, one row per policy worth
running; the row number is the PICK/SERVE menu slot and the port is `7776 + row`.
`MODELS.tsv.example` here is a snapshot of the live box menu (24 rows, 09-16). The tracked
mirror `../MODELS.tsv` is the first 16 of those rows — `fetch_student.sh` appends to the box
file only, on purpose, so a dirty tracked file can never block `git pull` mid-session.

## The scripts

**Scene + menu**

- `snap.py` — grabs a live RealSense colour frame (2.5 s auto-exposure warm-up, falls back to
  the last recorded frame if the camera is busy) and writes `/tmp/snap.png` side by side with
  `refs/ref_<task>.png`. This is the file the tracked `../SNAP_ANY.sh` shells out to, so without
  it that launcher is broken on a fresh box. Run it before the first take of a task to confirm
  the scene matches what the policy was trained on.
- `fetch_student.sh` — `HF_TOKEN=... ./fetch_student.sh <hid_simft|hid_mt> <step> [label]`.
  Pulls a rental student checkpoint from the hub straight into `runs/<run>/student_<step>.pt`
  and appends the matching `MODELS.tsv` row (system=student). Run mid-session when a new
  student lands and you want it in the menu without editing the TSV by hand.

**pi0.5 / baseline install** (one-time per box, CPU + disk only, no GPU)

- `pi05venv_setup.sh` — minimal variant: a python 3.12 venv that `.pth`-links the phantom venv's
  torch, plus `lerobot==0.4.4` / `transformers==4.53.2`.
- `pi05_setup2.sh` — the full attempt: clean python 3.11 venv, torch cu130, the patched
  transformers model files from `tf_replace.tgz`, then the 20k pi0.5 checkpoint + tokenizer from
  the hub and the processor patch. **Left the adapter unimportable** (missing zarr and friends).
- `pi05_setup3.sh` — the continuation that actually worked, and the one to run: same venv, but
  it exposes the phantom venv's site-packages at *lower* priority than pi05venv's own, then does
  the checkpoint download, processor patch and digest. Finish with `./serve_bg.sh 5` (needs
  ~8 GB free GPU).

**Mirroring rig output to the hub** (all need `HF_TOKEN` in the environment; all read-only
against the rig, safe to run while the arm is idle)

- `pack_upload_deploy.sh` — the current one. Tars+zstds each un-mirrored deploy day into a single
  `dataset_v3_packed/deploy_<day>.tar.zst`, verifies the hub size against the local size, then
  deletes the local tar. Use this: the hub repo is at its 20k-file limit and rejects raw folders.
- `upload_old_deploy.sh` — the older per-file variant (`rollouts/deploy_<day>/`). Superseded by
  the packed form above; kept for the days already mirrored that way.
- `relay_and_upload.sh` — compute3-side relay: rsync the packed dataset from compute over the
  headscale tunnel into `packed_relay/`, then upload every file with per-file retries and a
  final missing-files check.
- `pipeline_uploader.py` — the companion watcher: uploads each tarball as soon as rsync finishes
  it (size matched against `relay_sizes.txt`) instead of waiting for the whole relay. Run it
  alongside `relay_and_upload.sh`; the relay's own upload phase then skips what it already sent.

**Offline analysis, run on the box GPU between rig sessions**

- `offline_openloop_eval.py` — samples real action chunks on held-out val windows exactly the way
  deploy does and compares them to the demo actions (magnitude ratio, direction cosine, head vs
  tail cosine, seed variance, gripper MAE). Answers "is the model bad, or is deploy shifting the
  distribution?" — the flow velocity loss never checked what Euler sampling actually emits.
- `conditioning_ablation.py` — same-seed knockouts (prev-chunk zeroed, empty text, wrist zeroed,
  tactile zeroed, video greyed) reporting mean |Δaction| in mm, plus a best-of-4-seed direction
  cosine as a multimodality check. Answers "which inputs does the head actually listen to?".
- `esuite.sh` … `esuite7.sh` — the 08-28/29 replay battery, kept as the recipe rather than as
  something to re-run verbatim (checkpoints and episode ids are pinned to that week). They chain
  by polling each other's log for `ALL DONE<n>`, so launching one queues the rest:
  `esuite` (E0/E2/E3 levers sweep) → `esuite2` (E0 re-run, 16 seeds on the two slow episodes) →
  `esuite5` (replay with deploy's LoRA fold) → `esuite6` (`replay_deploy_path.py` parity, merge
  vs no-merge) → `esuite3` (replan-latency profile across NFE × k-seeds, incl. `--compile`) →
  `esuite4` (NFE 50 bimodality check on one episode). `esuite7` is standalone: deploy-RNG parity
  for v5 and v4. Each writes `runs/replay/<tag>.{json,log}` under `phantom/`.

## The standard session

1. **Serve** — terminal A: `./serve_bg.sh <rows...>` (or `./SERVE.sh` for a single model held in
   the foreground) to hold the menu models warm, one policy server per row on port `7776 + row`.
   `./serve_bg.sh status` shows what is listening.
2. **Launch** — terminal B: `./PICK.sh <row|label>` for a single take (model × inference preset →
   confirm → `GO_ANY.sh`), or `./EXPERIMENT.sh run <stage>` to warm a whole paper block and walk
   its cell plan automatically. Either way the launcher attaches to the already-warm server by
   checkpoint sha, so there is no per-take model load.
3. **Verdict** — answer the per-episode verdict prompt; takes land in
   `data/episodes/deploy/<day>[_experiment]/`, and `tools/rig/analysis/` turns them into the
   tables.

Before the first take of a task, `./SNAP_ANY.sh <task>` (which runs `snap.py`) to check the scene
against the training reference. `./serve_bg.sh stop <row>` only ever touches our own menu ports —
the box is shared, so never kill a process you did not start there.
