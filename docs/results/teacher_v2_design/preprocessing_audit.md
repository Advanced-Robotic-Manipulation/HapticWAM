# Teacher input preprocessing audit — 7 September 2026

**PASS for the three teachers in the frozen `teacher_robustness_v2_delivery` study.** No missing baseline subtraction or RGB/field conversion mismatch was found. This was a read-only CPU inspection of existing checkpoint metadata, current server metadata and saved observations; no checkpoint was reloaded, no inference or hardware operation was performed, and the active study was not changed. [Compact evidence](preprocessing_audit.json) records exact file identities, known hashes, source line references and the limits of this check.

| Checkpoint | Saved `wrench_baseline_rows` | Native effective value | Current server value | Resolved `mask_wrist` |
|---|---|---:|---:|---|
| ftA `teacher_001500.pt` | Absent | 0 | 0 | false |
| ftA `teacher_003000.pt` | Absent | 0 | 0 | false |
| Earlier teacher `v5_6.pt` | Absent | 0 | 0 | false |

All three use EMA and the same saved normalizers, canonical SHA-256 `42153b79ceb6a6c711296323c6b171e42929eb5dfe4018198d63d2cd3ace8c48`. Full checkpoint hashes remain in the JSON and [original CPU payload audit](teacher_payloads.json), inspected on 6 September at 22:57:09 UTC. Five server metadata files were read across the active study's first two start blocks; all report zero baseline rows and the matching identities/normalizers. Server JSON and observation file hashes were not collected; exact read paths are retained rather than inventing digests.

## What the baseline setting means

`phantom/train/common.py:941` resolves an absent `configs.train.wrench_baseline_rows` to zero for pre-v6 checkpoints. The setting controls the leading-row median subtracted from each **tactile pad** wrench in `contact_state` and `cpk_wrench` (`phantom/data/windows.py:90,373,430`). It does not request per-episode subtraction from `arm_ft`.

Training resamples the raw wrist stream and applies saved channel-wise `(value − mean) / std` (`windows.py:309`; `phantom/data/schema.py:154`). Native inference does the same (`phantom/inference/policy.py:109`), and native/simulator snapshots retain the raw wrist window. Safety-monitor baseline adaptation is a separate guard calculation. Subtracting the recorded initial wrist bias from the model input would change the trained convention for these teachers.

For canonical start 5928, the normalized initial wrist vector is approximately `[0.933, 1.083, −1.557, −1.922, 1.394, 0.477]`. Nine saved initial observations were checked: their 31×6 wrist windows preserve the declared raw initial bias within float32 rounding, maximum absolute difference `5.395e-7` from the float64 values. The two checked start-5963 observations use that start's different recorded bias, not the canonical one.

## Images, fields and measured residuals

The native camera requests `rgb8` (`phantom/drivers/real/realsense.py:73`). Isaac passes RGBA's first three channels as RGB; BGR conversion is confined to media output. Training and inference share `bilinear_resize` and image scaling `/127.5 − 1`. Grayscale gel images are repeated into three channels. Field keyframes receive one saved channel-wise normalization, while the 11-element `contact_state` remains raw apart from the optional tactile baseline subtraction, disabled here.

Observed shapes/dtypes match this contract: RGB `480×640×3 uint8`, wrist `31×6 float32`, gels `2×288×384 uint8`, fields `2×144×192×8 float32`, contact state `2×11 float32`. For each teacher's canonical seed-903101 first snapshot, both pads' gel images, keyframes after float32 casting, and `contact_state[:6]` are **exactly equal** to the declared Aug22 measured baseline. Both initial contact areas are zero. The measured residuals are preserved; this audit does not certify no contact elsewhere on the robot or calibrated physical force transfer.

## Reproduction and scope

On compute3, read the paths listed in `actually_read_server_metadata` and `actually_read_observations` in the JSON. With copies of this JSON and `teacher_payloads.json`, a CPU-only check is:

```python
import json
from pathlib import Path
import numpy as np

audit = json.loads(Path("preprocessing_audit.json").read_text())
baseline = np.load(audit["baseline"]["path"], allow_pickle=False)
for row in audit["actually_read_server_metadata"]:
    server = json.loads(Path(row["path"]).read_text())
    for key in ("checkpoint_sha256", "wrench_baseline_rows", "normalizers_sha256", "weights"):
        assert server[key] == row[key]
for row in audit["actually_read_observations"]:
    obs = np.load(row["path"], allow_pickle=False)
    info = json.loads(Path(row["policy_info_path"]).read_text())
    bias = np.asarray(info["initial_recorded_wrist_bias"])
    assert np.max(abs(obs["wrist_window"] - bias)) < 1e-6
    if row["exact_measured_baseline_check"]:
        for finger, side in enumerate(("left", "right")):
            assert np.array_equal(obs["gel"][finger], baseline[side + "_infer_img"])
            assert np.array_equal(obs["fields"][finger], baseline[side + "_keyframes"].astype(np.float32))
            assert np.array_equal(obs["contact_state"][finger, :6], baseline[side + "_wrench"].astype(np.float32))
```

The JSON pins the frozen protocol/source identity and gives each inspected source line and current local file hash. This is an input-contract audit, not model-performance evidence or physical sensor calibration. **Future limitation:** `SimulationPolicyAdapter` does not implement nonzero checkpoint tactile `wrench_baseline_rows`. A future checkpoint requiring it needs explicit support or rejection before use; none of the three active teachers requests it. No frozen source, model settings, observations or trial scores were edited.
