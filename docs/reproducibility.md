# Reproducibility map — paper ↔ released artifacts

Every model, episode set, table, figure and protocol constant in
**[arXiv:2609.23888](https://arxiv.org/abs/2609.23888)** (v1, 20 Sep 2026), mapped to the exact
public artifact it came from, with the command that fetches it and the command that re-derives
the number.

Every row below was checked against the hub, and every number marked **re-derived** was
recomputed from the published data before this file was written. Where the recomputation
disagrees with the paper, the disagreement is written down in
[§8 Known discrepancies](#8-known-discrepancies) rather than smoothed over.

**Conventions.** `hf` is the current Hugging Face CLI (`huggingface-cli` on older installs); no
token is needed, all repos are public. Byte sizes and `sha256` are the values the hub itself
reports: for model weights they are the **LFS sha256** listed in each model repo's
`index.jsonl`; for files in `hapticwam-evidence` they are the `sha256` column of that repo's
`MANIFEST.tsv`. Only the first 16 hex characters are printed here — enough to pin a file,
short enough to read. Verify a download with `shasum -a 256 <file>`.

---

## Contents

1. [Models — every arm the paper names](#1-models)
2. [Datasets and episode counts](#2-datasets-and-episode-counts)
3. [Tables II and III — rig results and pinch force](#3-tables-ii-and-iii)
4. [Protocol constants and metric definitions](#4-protocol-constants-and-metric-definitions)
5. [Table IV — the contact-imagination ablation](#5-table-iv)
6. [§IV-F offline probe, §IV-C parameter counts and latencies](#6-offline-probe-parameter-counts-and-latencies)
7. [Figures, and the reproduction path](#7-figures-and-the-reproduction-path)
8. [Known discrepancies](#8-known-discrepancies)
9. [What is NOT released, and why](#9-what-is-not-released-and-why)

---

## 1. Models

The paper deploys four policies. Each resolves to one file (or one `pretrained_model/`
directory) on the hub. `armteam/hapticwam-{teacher,student,baselines,ablations}` are
**Apache-2.0** and each carries an `index.jsonl` listing every file with its size and LFS
sha256.

| Paper arm | Repo | Path | Bytes | sha256 (first 16) |
|---|---|---|---|---|
| **HapticWAM: Teacher** (Tables I–III) | `armteam/hapticwam-teacher` | `teacher_v6_simft/teacher_002000.pt` | 393,116,309 | `1df9fc93510cf566` |
| ↳ its init: the 20,000-update teacher before the sim fine-tune (Table I, "20,000 demonstration updates") | `armteam/hapticwam-teacher` | `teacher_v6/teacher_020000.pt` | 393,116,245 | `4812cfbf012781f0` |
| **HapticWAM: Student** (Tables I–IV) | `armteam/hapticwam-student` | `hid_simft/student_001000.pt` | 372,335,337 | `f457edfe1b05a3b8` |
| **π₀.₅** baseline (§IV-C, Tables II–III) | `armteam/hapticwam-baselines` | `pi05_phantom_expert_v1/020000/pretrained_model/` — `model.safetensors` | 7,473,096,344 | `d27f4d92ce6257e8` |
| **Diffusion Policy** baseline (§IV-C, Tables II–III) | `armteam/hapticwam-baselines` | `diffusion_100k/pretrained_model/` — `model.safetensors` | 1,052,080,348 | `3a5ebefac12e1add` |
| Frozen backbone (§III-A, §IV-C) | `nvidia/Cosmos-Predict2.5-2B` | `robot/action-cond/38c6c645-7d41-4560-8eeb-6f4ddc0e6574_ema_bf16.pt` | — | upstream |
| Prompt-embedding cache (a required *input*, not a checkpoint) | `armteam/hapticwam-teacher` | `text_embeddings.pt` | 822,087,421 | `bef6ad98d2925799` |

```bash
hf download armteam/hapticwam-teacher  teacher_v6_simft/teacher_002000.pt --local-dir runs/teacher
hf download armteam/hapticwam-teacher  text_embeddings.pt                 --local-dir data/phantom-episodes/tasks
hf download armteam/hapticwam-student  hid_simft/student_001000.pt        --local-dir runs/hid_simft
hf download armteam/hapticwam-baselines --include 'pi05_phantom_expert_v1/020000/pretrained_model/*' --local-dir runs/baselines
hf download armteam/hapticwam-baselines --include 'diffusion_100k/pretrained_model/*'                --local-dir runs/baselines
```

### Ablation arms

The paper's only ablation (Table IV) is a **deployment-time intervention on the deployed student
checkpoint**, not a separate set of weights: the same `hid_simft/student_001000.pt` is run with
its generated contact frames clamped to zero. There is therefore **no "ablation checkpoint" to
download** — the arm is a flag, `run_deploy --null-imagination contact_zero` (see §5).

`armteam/hapticwam-ablations` (26 GB) holds every training arm that the paper does **not**
report — the v5 lineage, the video-loss study (`teacher_v6_ft_video1p0`, `_noVideoLoss`,
`_videoAttend`, `_control`), the multitask teacher and student (`teacher_v6_simft_multitask`,
`hid_mt`), the no-distillation and vision-only Cosmos controls (`cosmos_nodistill_v1`,
`cosmos_visiononly_v1`), the DAgger-round-2 students (`hid_r2_ftA`, `hid_2k_ftA`, `hid_v6_r2`)
and the π₀.₅ 60k resume. It also holds the complete offline evaluation sweeps (`eval_v6/`,
`eval_r2/`, `eval_mt/`, `eval_abl/`, `eval4/`, `eval_0906/`). None of these is cited in the
published paper; they are released so the selection of the deployed rungs is auditable.
Likewise `armteam/hapticwam-baselines/xvla_20k/` is an X-VLA baseline that was trained and
swept but does **not** appear in the published tables.

### Parameter counts (§IV-C)

| Paper | Re-derived | Verdict |
|---|---|---|
| π₀.₅ "updates 430.1 M parameters within a 3.62 B-parameter model" | total **3,616,757,520** = 3.62 B; action expert + time/action projections = 427,932,672 + 1,049,600 + 1,049,600 + 33,792 + 32,800 = **430,098,464** = 430.1 M | **match** |
| Diffusion Policy "trains all 263.0 M parameters from scratch" | **263,013,769** = 263.0 M (213 F32 tensors) | **match** |
| Teacher 27.9 M / student 26.4 M trainable | not re-derived — the counts are of *trainable* modules (LoRA + encoders + ACC + heads), which requires instantiating the model, not reading the checkpoint | not checked |

```bash
# reads only the safetensors header (a few KB), not the weights
python - <<'PY'
import json, struct, collections, httpx
for name, url in [
  ("pi0.5","https://huggingface.co/armteam/hapticwam-baselines/resolve/main/pi05_phantom_expert_v1/020000/pretrained_model/model.safetensors"),
  ("DP",   "https://huggingface.co/armteam/hapticwam-baselines/resolve/main/diffusion_100k/pretrained_model/model.safetensors")]:
    n = struct.unpack("<Q", httpx.get(url, headers={"Range":"bytes=0-7"}, follow_redirects=True).content[:8])[0]
    hdr = json.loads(httpx.get(url, headers={"Range":f"bytes=8-{7+n}"}, follow_redirects=True).content)
    g = collections.Counter()
    for k, v in hdr.items():
        if k == "__metadata__": continue
        p = 1
        for s in v["shape"]: p *= s
        g[".".join(k.split(".")[:4])] += p
    print(name, "total", sum(g.values()))
    for k, v in g.most_common(6): print("   ", k, v)
PY
```

---

## 2. Datasets and episode counts

| Paper statement (§IV-B, §IV-C) | Where it resolves | Hub's own number | Verdict |
|---|---|---|---|
| 3 tasks × (250 nominal + 35 deliberate-failure) = **285 per task** | `armteam/hapticwam-teleop-dataset`, `manifests_v6.tar` → `manifests/all.jsonl`; also `splits/paper_tasks_855.jsonl` | Carton 220 train + 30 val nominal + 35 fail; egg 223 + 27 + 35; waffles 220 + 30 + 35 | **match** |
| **855 in total** | same | **855** rows for the three evaluated tasks | **match** |
| split into **761 training and 94 validation** | same | **768 train / 87 val** | **mismatch** — see §8.1 |
| teleoperation corpus behind the teacher | `armteam/hapticwam-teleop-dataset`, `MANIFEST_OF_RECORD.md` | **1,115** episodes over **four** tasks (the 855 above + 260 whiteboard) | the paper never mentions whiteboard — see §8.2 |
| baselines trained on "the same **876-episode** baseline export, excluding deliberate failures" | LeRobot export produced by `tools/export_lerobot.py`; provenance table in [docs/pi05_baseline.md](pi05_baseline.md) | 991 four-task train rows − 115 deliberate-failure demos = **876** (177,193 frames); val 124 with 0 skips | count matches, **scope does not** — see §8.2 |
| sim fine-tune: **99 scripted simulation episodes** (98 completed placements, 1 ending with the object still gripped) and **203 robot rollouts**, **2,088 windows** | `armteam/hapticwam-sim-episodes` (`sim_expert_20260912/`, `sim_expert_20260914/`), `armteam/hapticwam-rollouts` | both repos are published with per-trial `index.jsonl`, but neither exposes a 99-episode or 203-rollout selection list | **not verifiable from the release as it stands** — see §9 |
| student dataset: **860 episodes** = 761 demonstrations + 99 rollouts (66 teacher, 33 student), 8 windows each → **6,880 windows** | `armteam/hapticwam-rollouts` (`rollouts_0901_0904.tar.zst`, `rollouts_0908_0909.tar.zst`, `manifests_r3.tar` = 1,238 rows = 1,115 teleop + 123 rollouts) | 860 × 8 = 6,880 is internally consistent; the 66/33 split is not tabulated in the release | arithmetic **match**, selection **not verifiable** |
| rig evaluation: **200 robot trials** (20 waffle + 20 carton + 10 egg × 4 models) | `armteam/hapticwam-rig-episodes`, `20260915_experiment/` | **200** episode directories; 50 per arm (`label:` = `v6_simft2k`, `stu_simft_001000`, `pi05`, `dp`); 80 Carton, 80 waffles, 40 egg | **match** |
| 94 held-out validation episodes used by the offline probe (§IV-F) | `armteam/hapticwam-evidence`, `rig_0916/probe/*.json` | each probe JSON reports `n_episodes = 124` over **four** tasks | **mismatch** — see §8.1 |

```bash
# the settled split lists, published alongside the manifests
hf download armteam/hapticwam-teleop-dataset --include 'splits/*' --repo-type dataset --local-dir .
#   splits/paper_tasks_855.jsonl          the three evaluated tasks
#   splits/additional_whiteboard_260.jsonl the fourth task, in the teacher's training mix
#   splits/deployed_teacher_train_991.txt  what the deployed teacher actually trained on
#   splits/deployed_teacher_val_124.txt    what the offline probe actually scored

# or recount from the manifest of record
hf download armteam/hapticwam-teleop-dataset manifests_v6.tar MANIFEST_OF_RECORD.md --repo-type dataset --local-dir .
python - <<'PY'
import tarfile, json, collections
rows = [json.loads(l) for n in ["all.jsonl"]
        for l in tarfile.open("manifests_v6.tar").extractfile(n).read().decode().splitlines() if l.strip()]
three = {"waffles","waffles_fail","Carton","Carton_fail","egg","egg_fail"}
sel = [r for r in rows if r["task"] in three]
print("all:", len(rows), dict(collections.Counter(r["split"] for r in rows)))
print("three evaluated tasks:", len(sel), dict(collections.Counter(r["split"] for r in sel)))
PY
```

### The rig takes behind the results tables

| Directory in `armteam/hapticwam-rig-episodes` | Episodes | Role |
|---|---|---|
| `20260915_experiment/` | **200** | the analysis set — every number in Tables II and III |
| `20260915_experiment_extra/` | 80 | other arms recorded in the same sessions (`stu_ftA_r2`, `stu_mt_001000`, `v6_simft_mt1500`); not in the paper |
| `20260915_experiment_superseded/` | 15 | takes replaced by a re-run under the same cell; kept for provenance |

Each episode directory holds one zarr group per stream plus `meta.json` and `planner_trace.json`.
Arms are identified by the `label:` tag, cells by the `seed:` tag (`seed = 100 + cell`).

---

## 3. Tables II and III

Both tables, and every descriptive count in §IV-D and §IV-G, come from three per-trial CSVs in
`armteam/hapticwam-evidence`. They are the scored form of the 200 episodes above.

| File | Bytes | sha256 (first 16) | Rows |
|---|---|---|---|
| `rig_0916/per_take_final.csv` | 47,339 | `7ab6ab43a69748bb` | 160 (waffles + Carton) |
| `rig_0916/per_take_egg.csv` | 12,183 | `7a9b37ae51caa792` | 40 (Egg) |
| `rig_0916/summary_by_arm.csv` | 1,319 | `bb0ba42264599245` | the per-arm rollup for waffles and Carton |

```bash
hf download armteam/hapticwam-evidence --repo-type dataset --local-dir . \
    --include 'rig_0916/per_take_final.csv' \
              'rig_0916/per_take_egg.csv' \
              'rig_0916/summary_by_arm.csv'
```

One field in `per_take_final.csv` was edited before publication: three rows carried a person's
name in the free-text `notes` column, recording whose call a late verdict was entered on, and
the name was removed. No measurement, verdict, class or count was changed, and the commands
below reproduce the published numbers from the edited file. That edit is why the `sha256` above
differs from the copy held in the project's internal archive.

The scoring rules those columns encode live in
[`tools/rig/analysis/final_stats.py`](../tools/rig/analysis/final_stats.py) (per-arm rollup,
which classes count as a sensor-defined placement, which column is the pinch peak) and
[`tools/rig/analysis/grasp_events.py`](../tools/rig/analysis/grasp_events.py) (the per-take
labelling itself, from the recorded fingertip streams). The paired sign test and the bootstrap
intervals are [`phantom/eval/stats.py`](../phantom/eval/stats.py).

### One command that re-derives Tables II and III

```bash
python - <<'PY'
import csv, statistics as st
B = "rig_0916/"
ARMS = [("v6_simft2k","Teacher"), ("stu_simft_001000","Student"), ("pi05","pi0.5"), ("dp","DP")]
rows = []
for f, fixed in (("per_take_final.csv", None), ("per_take_egg.csv", "Egg")):
    for r in csv.DictReader(open(B + f)):
        low = r["episode"].lower()
        r["task"] = fixed or ("Carton" if "carton" in low else "waffles")
        # operator placement verdict: 's', or 'crushed' (placed, too much force)
        r["op"] = str(r["operator_placed"]).lower() == "true" or r["verdict"] == "crushed"
        r["sensor_placed"] = r["class"] in ("placed_clean", "placed_crushed")
        rows.append(r)

print("TABLE II  operator placement verdicts")
for a, name in ARMS:
    out, tot, n = [], 0, 0
    for t in ("waffles", "Carton", "Egg"):
        g = [r for r in rows if r["arm"] == a and r["task"] == t]
        p = sum(r["op"] for r in g); tot += p; n += len(g)
        out.append("%3.0f%% (%2d/%2d)" % (100*p/len(g), p, len(g)))
    print("  %-8s %s   pooled %2d/%d = %.0f%%" % (name, "  ".join(out), tot, n, 100*tot/n))

print("\nTABLE III  pinch force at sensor-defined placements (N), mean +- sd (samples)")
for a, name in ARMS:
    out = []
    for t in ("waffles", "Carton", "Egg"):
        v = [float(r["pad_peak_pinch_n"]) for r in rows
             if r["arm"] == a and r["task"] == t and r["sensor_placed"] and r["pad_peak_pinch_n"]]
        out.append("--  (0)" if not v else
                   "%.1f  (1)" % v[0] if len(v) == 1 else
                   "%.1f+-%.1f (%2d)" % (st.mean(v), st.stdev(v), len(v)))
    print("  %-8s W %-16s C %-16s E %-16s" % (name, *out))
PY
```

Output, against the paper:

| Table II | Waffles | Carton | Egg | of 50 |
|---|---|---|---|---|
| Teacher — paper / re-derived | 70% (14/20) / **same** | 60% (12/20) / **same** | 40% (4/10) / **same** | 30 / **30** |
| Student — paper / re-derived | 95% (19/20) / **same** | 85% (17/20) / **same** | 50% (5/10) / **same** | 41 / **41** |
| π₀.₅ — paper / re-derived | 20% (4/20) / **same** | 30% (6/20) / **same** | 0% (0/10) / **same** | 10 / **10** |
| DP — paper / re-derived | 10% (2/20) / **same** | 15% (3/20) / **same** | 40% (4/10) / **same** | 9 / **9** |

Derived headline figures, all re-derived and matching: student **76.67 %** per-task mean
(paper "77 %") and **82.0 %** pooled; teacher **56.67 %** (paper "57 %") and **60.0 %** pooled.

| Table III | Waffles | Carton | Egg |
|---|---|---|---|
| Teacher — paper / re-derived | 16.7±4.7, n=13 / **16.7±4.7, 13** | 11.4±1.4, n=12 / **11.4±1.4, 12** | 24.9±9.2, n=5 / **24.9±9.2, 5** |
| Student | 14.3±3.6, n=19 / **14.3±3.6, 19** | 11.8±2.2, n=16 / **11.8±2.2, 16** | 18.8±9.1, n=6 / **18.8±9.1, 6** |
| π₀.₅ | 13.4±2.0, n=5 / **13.4±2.0, 5** | 16.0±4.9, n=5 / **16.0±4.9, 5** | — , n=0 / **—, 0** |
| DP | 14.1, n=1 / **14.1, 1** | 10.1, n=2 / **10.1 (sd 0.05), 2** | 27.1±3.1, n=4 / **27.1±3.1, 4** |

(The paper states that "the single-sample and two-sample DP summaries retain only the mean", so
the omitted 0.05 N carton sd is deliberate, not a gap.)

---

## 4. Protocol constants and metric definitions

| Constant / definition | Paper | Where it is fixed in the release | Re-derived |
|---|---|---|---|
| Success rate, Eq. (29) | binary operator-reviewed placement | `verdict == 's'` in the per-take CSVs, plus `verdict == 'crushed'` (placed, too much force); `phantom/eval/stats.py` ordinal level 3 | see §3 |
| Force reference, Eq. (30): `F_ref = μ_P + 3 s_P = 24.8 N` over the **73** waffle/carton trials the sensor rule scores as placements, pooled over all four models | 24.8 N, \|P\| = 73 | `class ∈ {placed_clean, placed_crushed}`, column `pad_peak_pinch_n` | **\|P\| = 73, μ = 13.650, s = 3.720, μ+3s = 24.81 N** — match |
| Over-grasp = pinch peak above `F_ref` | 4 exceedances: 2 teacher on waffles, 2 π₀.₅ on cartons; none for the student | column `over_grasp` | **4: teacher 2/0/0, student 0/0/0, π₀.₅ 0/2/0, DP 0/0/0 (waffles/carton/egg)** — match; the four trials are `ep_teacher_waffles_1789501367_007` (27.78 N) and `ep_teacher_waffles_1789507661_000` (25.42 N) on the teacher arm, `ep_student_Carton_1789493856_009` (46.75 N) and `ep_student_Carton_1789494357_009` (29.00 N) on the π₀.₅ arm. Episode directory names carry the recorder's `--system` (`teacher`/`student`), not the arm; the arm is the `label:` tag and the `arm` CSV column |
| Under-grasp = contact without a held lift | teacher 5/1/2, student 0/2/1, π₀.₅ 7/3/4, DP 10/1/1 | column `under_grasp` | **identical** |
| Egg damage scored by shell cracking | 2 cracks for DP, none for the others | `verdict == 'crushed'` in `per_take_egg.csv` | **2 DP egg trials, cells 104 and 105; 0 for every other arm** — match |
| Sensor rule vs operator agreement | 154 of the 160 waffle and carton trials | `op == sensor_placed` over `per_take_final.csv` | **154 / 160** — match. (Note the CSV's own `agree` column reports 148/160; it encodes a stricter agreement that also covers the grasp-quality label, not the placement verdict the paper quotes.) |
| Cells / seeds | seeds 101–120 (waffles, carton), 101–110 (egg), shared across models, model order alternated | `seed:` tag on every episode = `100 + cell`; `cell` column in the CSVs | **20 cells × 8 rows for waffles+carton, 10 cells × 4 rows for egg** — match |
| Paired test | exact two-sided sign test on a four-level ordinal (no held grasp / held / lifted / placed), ties dropped | `phantom.eval.stats sign_test` | see below |
| Wilson 95 % intervals | 76.4–99.1 (waffles), 64.0–94.8 (carton), 23.7–76.3 (egg) | standard Wilson score interval on 19/20, 17/20, 5/10 | **76.4–99.1, 64.0–94.8, 23.7–76.3** — match |

### Sign tests

The authoritative command reads the episode streams, so it needs the rig episodes downloaded:

```bash
hf download armteam/hapticwam-rig-episodes --repo-type dataset --local-dir data/rig \
    --include '20260915_experiment/*'
python -m phantom.eval.stats pairs data/rig/20260915_experiment \
    --arm-a label:pi05 --arm-b label:stu_simft_001000 --task waffles \
    --hw configs/hardware.nuc.yaml
```

A CSV-only approximation of the same ordinal (`placed` → 3, else `grasp_success` → 2, else a
held-but-unlifted class → 1, else 0) reproduces the paper's p-values:

| Comparison | Paper | Re-derived (CSV ordinal) |
|---|---|---|
| student vs teacher, waffles / carton / egg | 0.062, 0.125, 1.0 | **0.0625, 0.1250, 1.0000** — match |
| student vs π₀.₅, every task | p ≤ 0.016 | **0.0005 (waffles), 0.0063 (carton), 0.0078 (egg)** — the bound holds; the paper's 0.016 is conservative relative to this reconstruction |
| student vs DP, waffles and carton | p < 0.001 | **0.0001, 0.0001** — match |
| student vs DP, egg | 0.69 | **0.6875** — match |

---

## 5. Table IV

The contact-imagination ablation compares the intact student against the same checkpoint with
its generated contact frames clamped to a zero contact package, on the first ten starts per task.

| Column | Paper | Status |
|---|---|---|
| Intact student: 90 % (9/10) waffles, 70 % (7/10) carton, 50 % (5/10) egg, 70 % (21/30) | — | **re-derived and matching** — it is the cells 101–110 subset of the student arm in `20260915_experiment/`, i.e. the same released trials as Table II |
| Zero contact: 10 % (1/10), 30 % (3/10), 0 % (0/10), 13.3 % (4/30) | — | **not re-derivable from the release**: those 30 episodes are deliberately not published (§9). The reported outcomes stand as measured; this file does not attempt to confirm them |
| Absolute reduction 56.7 percentage points | 70 − 13.3 | arithmetic consistent |

```bash
# the intact column, from the released trials
python - <<'PY'
import csv
B = "rig_0916/"
rows = []
for f, fixed in (("per_take_final.csv", None), ("per_take_egg.csv", "Egg")):
    for r in csv.DictReader(open(B + f)):
        low = r["episode"].lower()
        r["task"] = fixed or ("Carton" if "carton" in low else "waffles")
        rows.append(r)
tot = 0
for t in ("waffles", "Carton", "Egg"):
    g = [r for r in rows if r["arm"] == "stu_simft_001000" and r["task"] == t
         and 101 <= int(r["cell"]) <= 110]
    p = sum(str(r["operator_placed"]).lower() == "true" or r["verdict"] == "crushed" for r in g)
    tot += p
    print("%-8s %d/%d" % (t, p, len(g)))
print("all tasks %d/30 = %.1f%%" % (tot, 100*tot/30))
PY
```

### Re-running the intervention yourself

The intervention is a deploy flag, and it is in the public code:

```bash
python -m phantom.scripts.run_deploy --system student \
    --ckpt runs/hid_simft/hid_simft/student_001000.pt \
    --task waffles --null-imagination contact_zero   # {none, prev_cpk, contact_zero}
```

`none` is the intact deployment condition, `contact_zero` clamps the generated contact frames,
`prev_cpk` zeroes only the previous-package input to ACC. Every episode is tagged
`null:<mode>` — including `null:none` — so an intervention arm and its intact partner are
distinguishable even though they load the same checkpoint
([`phantom/scripts/run_deploy.py`](../phantom/scripts/run_deploy.py),
[`phantom/inference/policy.py`](../phantom/inference/policy.py) `NULL_IMAGINATION_MODES`). The
pairing is then scored with `phantom.eval.stats pairs --arm-a null:none --arm-b
null:contact_zero`.

---

## 6. Offline probe, parameter counts and latencies

### The §IV-F endpoint-error probe

Four numbers, three of them in published JSONs and the fourth a column of the same rows.

| Paper | Artifact in `armteam/hapticwam-evidence` | sha256 (first 16) | Re-derived |
|---|---|---|---|
| intact student **22.31 mm** | `rig_0916/probe/student_none.json` (`summary.null = "none"`, `is_deploy_condition: true`) | `45704557d13e7e4a` | **22.309** |
| contact frames clamped to zero **28.31 mm** | `…/probe/student_contact_zero.json` | `92e054ac98b9dacd` | **28.311** |
| ACC input zeroed **22.35 mm** | `…/probe/student_prev_cpk.json` | `40c089c91ec30e58` | **22.348** |
| a policy that does not move **28.2 mm** | the `zero_endpoint_err_mm` column of the *same* rows — a per-window floor, not a separate run | — | **28.231** |
| (not in the paper) contact pinned to the ground-truth package | `…/probe/student_contact_gt.json` | `48df420ee4f51aaf` | 24.827 |

```bash
hf download armteam/hapticwam-evidence --repo-type dataset --local-dir . \
    --include 'rig_0916/probe/*'
python - <<'PY'
import json, statistics as st
B = "rig_0916/probe/"
for n, what in [("student_none","intact (deploy condition)"),
                ("student_contact_zero","contact frames clamped to zero"),
                ("student_prev_cpk","ACC previous-package input zeroed"),
                ("student_contact_gt","contact pinned to GT (not in the paper)")]:
    d = json.load(open(B + n + ".json")); r = d["rows"]
    print("%-22s %-38s windows=%d episodes=%d  endpoint %.2f mm   no-move floor %.2f mm"
          % (n, what, len(r), d["summary"]["n_episodes"],
             st.mean(float(x["endpoint_err_mm"]) for x in r),
             st.mean(float(x["zero_endpoint_err_mm"]) for x in r)))
PY
```

The producer is [`tools/terminal_eval.py`](../tools/terminal_eval.py); the metric key is
`endpoint_err_mm` and the floor is `zero_endpoint_err_mm`. To recompute from weights rather
than from the published JSON:

```bash
python tools/terminal_eval.py --ckpt runs/hid_simft/hid_simft/student_001000.pt \
    --data data/phantom-episodes/tasks --hardware configs/hardware.nuc.yaml \
    --nfe 1 --seeds 4 --split val --null none --out eval/student_none.json
#   … --null contact_zero  → 28.31 mm ;  --null prev_cpk → 22.35 mm
python -m phantom.eval.stats offline eval/student_none.json eval/student_contact_zero.json
```

Each run prints and stores `summary["null_semantics"]`, which states exactly what was clamped —
use it rather than the filename when reading a result.

**Caveat:** every one of these JSONs reports `n_episodes = 124` over four tasks (Carton 120,
egg 108, waffles 120, whiteboard 148 windows), not the 94 three-task episodes the paper's
sentence implies. See §8.1.

### Replan latency (§IV-C)

Per-replan latency is recorded by the deploy runtime into `planner_trace.json` inside every
episode directory (`phantom/deploy/planner.py` → `phantom/deploy/runtime.py`), and DP's
observation spacing is `diag["obs_dt_s"]` in the same rows
(`phantom/inference/lerobot_policy.py`). `phantom/eval/metrics.py::latency_stats` aggregates
mean and p95; the paper reports **medians**, for which no committed aggregator exists. The
numbers are re-derivable from the released episodes:

```bash
hf download armteam/hapticwam-rig-episodes --repo-type dataset --local-dir data/rig \
    --include '20260915_experiment/*/planner_trace.json' '20260915_experiment/*/meta.json'
python - <<'PY'
import json, glob, statistics as st, collections
lat, obs = collections.defaultdict(list), collections.defaultdict(list)
for ep in sorted(glob.glob("data/rig/20260915_experiment/ep_*")):
    m = json.load(open(ep + "/meta.json"))
    arm = next((t[6:] for t in m.get("tags", []) if t.startswith("label:")), "?")
    try: tr = json.load(open(ep + "/planner_trace.json"))
    except FileNotFoundError: continue
    for r in tr:
        if r.get("latency_s") is not None: lat[arm].append(float(r["latency_s"]))
        d = r.get("diag") or {}
        if d.get("obs_dt_s"): obs[arm].append(float(d["obs_dt_s"]))
for a in sorted(lat):
    print("%-20s median replan %.3f s (n=%d)%s" % (a, st.median(lat[a]), len(lat[a]),
          "   median obs_dt %.3f s" % st.median(obs[a]) if obs[a] else ""))
PY
```

All four medians reproduce:

| Arm | `label:` tag | Paper | Re-derived median |
|---|---|---|---|
| Teacher (compiled kernels) | `v6_simft2k` | 0.445 s | **0.445 s** (2,513–3,131 replans) |
| Student (uncompiled) | `stu_simft_001000` | 0.636 s | **0.636 s** (1,194–2,065 replans) |
| π₀.₅ | `pi05` | 0.127 s | **0.127 s** (1,453 replans) |
| Diffusion Policy | `dp` | 0.389 s | **0.389 s** (1,920 replans over all 50 DP episodes) |
| DP observation history spacing | `dp`, `diag.obs_dt_s` | 0.486 s | **0.500 s** median, 0.499 s mean (1,870 samples) — see §8.6 |

(Ranges reflect two passes at different download completeness; the medians are stable to
±0.001 s. The hub rate-limits bulk small-file reads, so fetch the traces with
`hf download --include` as above rather than one HTTP GET per episode.)

---

## 7. Figures, and the reproduction path

| Figure | What it shows | Source |
|---|---|---|
| Fig. 1 | modality overview and the three evaluated tasks | scene and gel frames from the teleop corpus (`armteam/hapticwam-teleop-dataset`) and the rig takes (`armteam/hapticwam-rig-episodes`); no derived quantity |
| Fig. 2 | framework overview | diagram, no data |
| Fig. 3 | HapticWAM internals (DiT blocks, ACC, HID) | diagram, no data; the modules are under `phantom/model/` — see [docs/code_structure.md](code_structure.md) |
| Fig. 4 | the physical setup | photographs of the rig |
| Fig. 5 | real setup (left) and the Isaac Sim scene (right) | the sim scene is `armteam/hapticwam-sim-episodes`; the scene assets are `assets/sim/` in this repository, with `PROVENANCE.json` |

No figure in the published v1 plots the rig numbers. The plotting script and its rendered
outputs are archived anyway, and they read the same per-take CSVs as §3, so they are a second,
independent check on Tables II and III:
`rig_0916/figures/{make_figures_0916.py, fig_rig_outcomes.{png,pdf}, fig_rig_pinch.{png,pdf}}`
in `armteam/hapticwam-evidence`.

### Provisioning scripts, and every hub id they touch

`tools/provision_v4.sh`, `provision_v5.sh`, `provision_distill.sh`, `provision_distill_v6.sh`,
`provision_dagger_r2.sh` and `provision_student_mt.sh` set up a rented GPU box from the hub with
nothing but an `HF_TOKEN`. Every hub reference they make was checked; all resolve:

| Script | Repo | Paths fetched | Status |
|---|---|---|---|
| `provision_v4/v5/distill` | `armteam/hapticwam-teleop-dataset` | `{Carton,egg,waffles,whiteboard}{,_fail}.tar.zst`, `manifests.tar`, `norm_stats.json`, `batch_20260822.tar.zst` | all present |
| `provision_v4/v5/distill` | `armteam/hapticwam-teacher` | `text_embeddings.pt` | present |
| `provision_v5/distill` | `armteam/hapticwam-ablations` | `teacher_v5_batch0822/teacher_003000.pt`, `teacher_v5_ftA/teacher_001500.pt` | present |
| `provision_v5/distill` | `armteam/hapticwam-teleop-raw` | `archive/` sessions | present (114 sessions) |
| `provision_v4/v5/distill` | `nvidia/Cosmos-Predict2.5-2B` | `robot/action-cond/38c6c645-…_ema_bf16.pt`, `robot/action-cond/cr1_empty_string_text_embeddings.pt` | present |
| `provision_distill_v6` | `armteam/hapticwam-teacher` / `-ablations` | `$V6_CKPT`, default `teacher_v6/teacher_020000.pt` | present |
| `provision_distill_v6` | `armteam/hapticwam-teleop-dataset` | `manifests_v6.tar` | present |
| `provision_student_mt` | `armteam/hapticwam-teacher` | `teacher_v6_simft/teacher_002000.pt` | present |
| `provision_student_mt` | `armteam/hapticwam-ablations` | `teacher_v6_simft_multitask/teacher_001500.pt` | present |
| `provision_dagger_r2` | `armteam/hapticwam-rollouts` | `rollouts_0901_0904.tar.zst`, `rollouts_0908_0909.tar.zst`, `manifests_r3.tar` | present |
| `provision_dagger_r2` | `armteam/hapticwam-ablations` | `teacher_v5_ftA/teacher_001500.pt` | present |

Two references are **absent by design and already documented in the scripts themselves**:
`hapticwam-ablations/teacher_v4_790eps/teacher_020000.pt` (the v4 control init, commented out
with a note saying the file was never copied over) and, in the same comment, the now-deleted
`armteam/phantom-checkpoints` repository. Neither is on a live code path. The same deleted
repository is still linked from the hub-side READMEs of `hapticwam-rig-episodes` and
`hapticwam-sim-episodes`, and appears in the `source` provenance column of
`hapticwam-rollouts/index.jsonl` — stale links to fix on the hub, not broken fetches.

The baseline export path is
[`tools/export_lerobot.py`](../tools/export_lerobot.py) → [`docs/pi05_baseline.md`](pi05_baseline.md),
validated by [`tools/validate_lerobot_export.py`](../tools/validate_lerobot_export.py).

---

## 8. Known discrepancies

These are differences between what the paper states and what the released artifacts show. They
are recorded here rather than corrected; the paper is not edited by this document.

### 8.1 The validation split: 94 vs 87 vs 124

The paper says the 855 three-task episodes are "split into 761 training and 94 validation
episodes", and that the offline probe scores "the 94 held-out validation episodes".

* The manifest of record (`manifests_v6.tar` → `manifests/all.jsonl`, published as
  `splits/paper_tasks_855.jsonl`) splits those 855 as **768 train / 87 val**.
* The manifest as a whole is **991 train / 124 val** over four tasks, and
  `armteam/hapticwam-teacher/teacher_v6/train_v6.log` records `manifest split 'train': 991
  episodes` — this is what the deployed teacher actually trained on.
* `phantom/train/common.py::manifest_split` takes the manifest's `split` column verbatim; no
  committed code re-splits, re-seeds or holds out sessions at load time. A filter can only
  shrink a split, never move episodes from train into val, so **761/94 is not reachable from
  the released manifest by any published code path**.
* The published probe JSONs report `n_episodes = 124`, so the endpoint-error numbers were
  measured on 124 four-task validation episodes, not on 94 three-task ones.

The arithmetic 761 = 855 − 94 suggests the paper's pair was formed by subtracting an evaluation
episode count from the three-task total. Either way, the episode counts in the release are the
ones the checkpoints were produced with.

### 8.2 The whiteboard task is in the training mix but not in the paper

The paper describes three tasks. The teleoperation corpus, the teacher's training split and the
876-episode baseline export all include a fourth task, **whiteboard** (260 episodes: 223 train
+ 37 val). `splits/additional_whiteboard_260.jsonl` isolates it. Consequently:

* "855 episodes" is correct for the three evaluated tasks, but the deployed teacher saw 1,115;
* "the same 876-episode baseline export, excluding deliberate failures" is correct as a count
  (991 four-task train rows − 115 deliberate-failure demos, per
  [docs/pi05_baseline.md](pi05_baseline.md)) but is a **four-task** set, not a three-task one.

### 8.3 "on eggs, every model exceeds it"

§IV-G says the force reference "is an in-sample waffle/carton band, not calibrated for eggs,
where every model exceeds it." Re-derived from `per_take_egg.csv`: the teacher exceeds 24.8 N on
4 of its 10 egg trials, the student on 3, DP on 3 — and π₀.₅ on **0**, because all ten of its
egg trials record a pinch peak of 0.00 N (it never closed on the egg; Table II gives it 0/10 and
Table III no force samples). The statement holds for the three models that made contact.

### 8.4 π₀.₅ per-task sign-test bound

The paper reports `p ≤ 0.016` for the student over π₀.₅ on every task. The CSV-ordinal
reconstruction gives 0.0005 / 0.0063 / 0.0078 — the bound holds, but no task attains 0.016. The
exact per-task values depend on how stages 0–2 are recovered (the authoritative route is
`phantom.eval.stats pairs --hw …` on the episode streams, which is stricter than the CSV proxy),
so the discrepancy is in the reconstruction, not necessarily in the paper.

### 8.5 Diffusion Policy observation spacing

§IV-C says "DP's deployment history is 0.486 s apart rather than the 0.1 s training interval."
Re-derived from `diag["obs_dt_s"]` over all 50 DP episodes (1,870 replans): **median 0.500 s,
mean 0.499 s**, and the same figure as a per-episode median-of-medians. The qualitative point —
the deployed history is ~5× the training interval — stands; the exact 0.486 s is 3 % below
anything this data yields, so it was probably computed over a different subset of replans.

### 8.6 Hub-side notes (not code)

* `hapticwam-rig-episodes` and `hapticwam-sim-episodes` READMEs link
  `armteam/phantom-checkpoints`, which no longer exists.
* `hapticwam-baselines/README.md` says `A_visiononly_nfe1.json` and `B_nodistill_nfe1.json` "are
  one baseline table in the paper" — the published v1 has no such table, and no X-VLA row.
* The published `train_config.json` of the π₀.₅ and DP baselines records the absolute training
  paths of the machine they were trained on.

---

## 9. What is NOT released, and why

Being explicit about the gaps is part of the map.

| Not released | Why | What you get instead |
|---|---|---|
| **The 30 zero-contact rig trials behind Table IV** | withheld by the authors; the reported outcomes stand as measured, they are simply not in the release | the intact column of Table IV **is** reproducible — it is the first ten cells per task of the released 200 trials (§5). For the intervention arm itself: the flag that produces it is public, `run_deploy --null-imagination contact_zero`, and it drives the same mechanics as the offline probe (`phantom/inference/policy.py`, `tools/terminal_eval.py`), so the arm can be re-run on a rig from the released student checkpoint |
| The exact 99-episode simulation subset and 203-rollout subset behind the teacher's sim fine-tune | no selection list was exported | both source repos are fully public (`hapticwam-sim-episodes`, `hapticwam-rollouts`); the fine-tune recipe is in §IV-C and `tools/provision_student_mt.sh` |
| The 66 teacher / 33 student split of the student's 99 rollout episodes | not tabulated | `manifests_r3.tar` in `hapticwam-rollouts` indexes all 123 rollout episodes with their policy provenance |
| Raw per-replan medians as a committed script | only mean and p95 are aggregated by `phantom/eval/metrics.py` | the raw `latency_s` and `obs_dt_s` are in every released episode's `planner_trace.json`; the aggregation command is in §6 |
| Teacher/student trainable-parameter counts as a printed artifact | they are properties of the instantiated model | build the model from the released checkpoint and config and count `requires_grad` parameters |
| Operator identities, raw session video, and the paper's source history | privacy and scope | the scored per-trial CSVs, the probe JSONs and the plotting script are published in full |
| The Cosmos-Predict2.5-2B backbone weights | third-party | `nvidia/Cosmos-Predict2.5-2B`, fetched by the provisioning scripts under NVIDIA's licence |

Everything else the paper depends on — the four deployed checkpoints, the 1,115-episode
teleoperation corpus, the simulation episodes, the rollouts, the 200 scored rig trials, the
per-trial CSVs, the offline probe JSONs and the scoring code — is public.
