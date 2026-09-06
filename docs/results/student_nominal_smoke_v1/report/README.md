# Report reproduction

`report_helpers.py` is the exact CPU helper identified by `summary.json`:

```text
SHA256 8cc4f2678ad6c3824482b4ff6237262385b242996b5cc0caf206f86e9d800834
```

From the bundle root, use Python 3 with NumPy installed:

```bash
python report/regenerate.py --root . --out /tmp/student-smoke-report
```

This regenerates `per_trial.csv` and `summary.json` from the frozen per-case scores, compact numeric/command logs and video manifest. It checks original scored-input hashes through the same `trial_row` helper. It does not run a model, change thresholds, read bulky observation arrays, or start hardware/simulation. Absolute provenance paths follow the mirror location; numerical results and relative video links remain the same. Relative video links in the exported CSV target the bundle's normal `report/` location.

The frozen analyzer source belongs to the accompanying simulator source package; its authoritative four-case outputs are retained under `campaign/analysis/`. Regenerating this presentation is separate from rescoring the physics traces.
