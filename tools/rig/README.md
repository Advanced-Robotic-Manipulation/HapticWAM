# tools/rig — the rig-box (RTX 5090) launcher scripts

Tracked copies of the box-local scripts that live in `~/phantom-icra-2027/` on the rig box
(the box that drives the UR3). The box copies are the live ones; keep them in sync with
this directory when either side changes.

The ordinary LEVERS menu plays 12 steps with no hold/release opt-ins. A teacher
pick/place qualification instead needs the full hardware file with the 2.5 s
verified hold and minimal_v5 release enabled, and 10 played steps.

- `PICK.sh` — interactive launcher: model menu (from `MODELS.tsv`) x inference preset
  (LEVERS / PLAIN / VETO / CUSTOM) -> confirm -> exec `GO_ANY.sh`.
- `MODELS.tsv` — the curated model menu (`label<TAB>ckpt-path-relative-to-phantom/<TAB>note`).
  Only list builds worth running; staging a new build = add a row (concrete .pt paths, not DEMO symlinks).
- `GO_ANY.sh` — the underlying launcher: preflight (arm ping, camera USB3), then
  `run_deploy --system teacher --ckpt $CKPT --ema ... $EXTRA`. `GO_<task>.sh` /
  `GO_v5_<task>.sh` are thin wrappers pinned to the v4 / staged-v5 DEMO.pt symlinks.
- `stage_v5.sh`, `SNAP_ANY.sh`, `GRIPPER_OPEN.sh`, `GRIPPER_RESET.sh`, `screenshot_*.sh` — ops helpers.
- `probe_rig.py` — read-only pre-session wiring probe (RTDE, Robotiq :63352, RealSense).
- `box/` — the operator scripts that only existed on the box: `snap.py` (the frame grabber
  `SNAP_ANY.sh` calls), `fetch_student.sh`, the pi0.5 baseline install, the hub mirroring
  scripts and the offline replay batteries. See `box/README.md`.

Session recipe + safety rules: `docs/rig_session_v5.md` (mirrored on the box as
`RIG_SESSION_V5_README.md`).
