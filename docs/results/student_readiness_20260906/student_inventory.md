# Student checkpoint readiness

The requested next pair is **revised ftA student (`student_ftA_r1`) and the v5_6 student**, each compared with the unchanged ftA teacher. Revised ftA is the newer student revision; v5_6 is an alternative teacher lineage, not a newer revision of ftA. The local registry selects both. No newer `hid_r1_v5_6` or `hid_r2` run appeared in the model archive metadata at revision `17e2821b7bfbcab54e08cd2f0bbb9859b9b88729`.

This audit only read existing checkpoints on CPU, model metadata and source. It launched no model, training or hardware process and downloaded no checkpoint. The [compact inventory](student_inventory.json) records current complete-file hashes; [raw metadata](student_inventory_raw.json) retains model/training configuration and normalization. The earlier [teacher inventory](teacher_inventory_raw.json) is background from before the user clarified “students.”

| Role | Path relative to `/home/physicalai/phantom-icra-2027/phantom` | Saved architecture | Completion / EMA |
|---|---|---|---|
| First student | `runs/student_ftA_r1/student_001200.pt` | Student | 1200/1200 steps; 630 finite EMA tensors |
| Second student | `runs/student_v5_6/student_001200.pt` | Student | 1200/1200 steps; 630 finite EMA tensors |
| Later original comparison | `runs/student_ftA/student_001200.pt` | Student | 1200/1200 steps; 630 finite EMA tensors |
| Later no-HID control | `runs/control_ftA/teacher_001200.pt` | **Student**, despite filename | 1200/1200 steps; 630 finite EMA tensors |
| Later no-HID control | `runs/control_v5_6/teacher_001200.pt` | **Student**, despite filename | 1200/1200 steps; 630 finite EMA tensors |
| Fixed teacher reference | `runs/teacher_v5_ftA/teacher_001500.pt` | Teacher | Selected step1500; finite EMA; historical run reached3000 |

The two primary file identities are:

```text
student_ftA_r1  b13362472843c4b09d9c7a091c0e97ff34a58ea45411961886f5d72ebf1703b3
student_v5_6    aee98c1cc01f06ba5b507422f433a7676c174603d92decf2e2c25e52dd857eca
```

All five student/control payloads reach their declared final training step, have finite EMA values, finite normalization and positive standard deviations. No active teacher/student training process was present at the initial inventory. Full student training logs were not present locally, so this establishes artifact readiness, not every training-update property or closed-loop reliability. The registry reports offline errors22.7mm for revised ftA,23.6mm for original ftA and27.3mm for v5_6; these are endpoint summaries, not task-success rates.

Revised ftA saves `teacher_nfe=-1` and `w_sigma=1.0`. Current `phantom/train/distill_hid.py:73–79,145–183` interprets these as using the teacher’s full configured sampling count and training the sigma readout. Original ftA and the v5_6 student are round0 payloads without those recorded revisions. Retain the original as a later revision comparison; it adds less information to the first pilot than the requested second lineage.

## Exact inference and input handling

For the declared matched follow-up, use the source-backed **PICK/LEVERS profile**: `--system student`, EMA, NFE1, K4, guidance1, persistent noise, parity fixes, live terminal veto, task text `waffles`, maximum10 playback steps. `tools/rig/PICK.sh:26–34,91` composes this with `GO_ANY.sh:13–16`. Set the experiment horizon explicitly; the historical PICK150s limit is separate from the previous60s campaign.

The checkpoint’s stored default is **NFE5**, with ACC `two_pass`; K4 is a runtime choice. The plain native CLI defaults to K1, no parity flag and no persistent-noise flag. Do not call NFE1/K4 a value encoded in the weights, and do not silently mix the LEVERS and plain GO_ANY recipes.

Load each checkpoint’s own model configuration and mean/std through `phantom/scripts/run_deploy.py:49–103`. Revised ftA uses `action_noise_per_strip=True` and `cond_dropout_p=0`; v5_6 student uses `False` and `.1`. Both use `rope_time_mode=time_true`, ACC `two_pass`, scene video enabled, and `mask_wrist=False`. Replacing these with a shared hand-written model configuration would change the model being evaluated.

All candidates and the ftA teacher share exactly the same normalization values, canonical SHA256 `42153b79ceb6a6c711296323c6b171e42929eb5dfe4018198d63d2cd3ace8c48`. The complete arrays remain in the raw inventory; there is no need to refit or borrow simulated statistics.

The registry’s “sensor-free” label needs qualification: these students are **free of observed fingertip tactile tokens but retain wrist force/torque**. `phantom/model/hht/hht.py:78–105` builds proprioceptive tokens from wrist and arm state before returning early for students. Since `mask_wrist=False`, the wrist window remains a model input. Physical tactile sensors also remain required by safety and the grip latch. A wrist-masked `vision_only`/`drop_tactile` mode is a distinct intervention, not an equivalent launch of these weights.

## Reproduce the read-only audit

Run the helper through stdin so no remote source file is created:

```bash
ssh compute3 'cd /home/physicalai/phantom-icra-2027/phantom && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -' \
  < inventory_students.py > student_inventory_raw.json
```

The helper hashes existing files, scans EMA values and captures saved configuration. It does not construct a model or allocate CUDA tensors. Runtime warmup/settings identity and the bounded simulation trials are separate work owned by the experiment runner.
