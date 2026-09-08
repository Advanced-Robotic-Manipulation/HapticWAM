# tools/rig — the compute3 (5090) rig launcher scripts

Tracked copies of the box-local scripts that live in `~/phantom-icra-2027/` on compute3
(the box that drives the UR3). The box copies are the live ones; keep them in sync with
this directory when either side changes.

For the **fixed ftA1500 teacher pick/place qualification**, follow
[the explicit main-based quickstart](../../docs/rig_teacher_fixed_inference.md).
It prepares the full hardware file with the 2.5 s verified hold and enables
minimal_v5 release. The ordinary LEVERS menu is a different recipe: 12 played
steps, without those opt-ins; the tested simulator candidate used 10.

- `PICK.sh` — interactive launcher: model menu (from `MODELS.tsv`) x inference preset
  (LEVERS / PLAIN / VETO / CUSTOM) -> confirm -> exec `GO_ANY.sh`.
- `MODELS.tsv` — the curated model menu (`label<TAB>ckpt-path-relative-to-phantom/<TAB>note`).
  Only list builds worth running; staging a new build = add a row (concrete .pt paths, not DEMO symlinks).
- `GO_ANY.sh` — the underlying launcher: preflight (arm ping, camera USB3), then
  `run_deploy --system teacher --ckpt $CKPT --ema ... $EXTRA`. `GO_<task>.sh` /
  `GO_v5_<task>.sh` are thin wrappers pinned to the v4 / staged-v5 DEMO.pt symlinks.
- `stage_v5.sh`, `SNAP_ANY.sh`, `GRIPPER_OPEN.sh`, `GRIPPER_RESET.sh`, `screenshot_*.sh` — ops helpers.
- `probe_rig.py` — read-only pre-session wiring probe (RTDE, Robotiq :63352, RealSense).

Session recipe + safety rules: `docs/rig_session_v5.md` (mirrored on the box as
`RIG_SESSION_V5_README.md`).
