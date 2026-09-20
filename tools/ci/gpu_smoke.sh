#!/usr/bin/env bash
# ============================================================================
# GPU smoke for a dependency bump (torch pins, cosmos, transformers, ...).
#
# CI (.github/workflows/ci.yml) proves the model still builds and runs on CPU.
# This script is the same proof ON A CUDA BOX, against the exact pins in
# requirements/*.txt — the thing CI cannot reach. Run it after changing a pin.
#
# It builds a THROWAWAY venv and NEVER touches ~/venvs/phantom.
# It is NOT a training run: tiny model, synthetic data, 1-2 optimizer steps,
# well under a minute of GPU compute.
#
#   bash tools/ci/gpu_smoke.sh                                # 5090 / rig pins
#   REQ=requirements/requirements-h100.txt bash tools/ci/gpu_smoke.sh
#   VENV=~/venvs/torch213_smoke bash tools/ci/gpu_smoke.sh    # keep the venv
#   FULL=1 bash tools/ci/gpu_smoke.sh                         # + the whole suite
#
# Needs: the cosmos-predict2.5 submodule checked out
#        (git submodule update --init --depth 1 cosmos-predict2.5).
# Does NOT need the 4.8 GB Cosmos-Predict2.5-2B weights (everything is --tiny).
#
# Leaves behind: runs/teacher/gpu_smoke_* and runs/hid/gpu_smoke_* (gitignored
# checkpoints, a few MB) plus one mock deploy episode under the data root.
# ============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

REQ="${REQ:-requirements/requirements-5090.txt}"
PY="${PY:-python3}"
VENV="${VENV:-$(mktemp -d)/gpu_smoke_venv}"
STAMP="$(date +%m%d_%H%M%S)"

echo "== repo         $REPO"
echo "== requirements $REQ"
echo "== venv         $VENV"
echo

if [ ! -f cosmos-predict2.5/cosmos_predict2/__about__.py ]; then
  echo "FAIL: the cosmos-predict2.5 submodule is not checked out." >&2
  echo "      git submodule update --init --depth 1 cosmos-predict2.5" >&2
  exit 1
fi

# --- 1. install the pinned stack into a throwaway venv ----------------------
"$PY" -m venv "$VENV"
"$VENV/bin/pip" install -q -U pip
"$VENV/bin/pip" install -r "$REQ"
"$VENV/bin/pip" install -e .

# --- 2. the pins resolved, the GPU is visible, and the WHEEL was built for
#        this card's compute capability (the whole point of the cu-index pin)
echo
echo "===== [1/6] import + device check ====="
"$VENV/bin/python" - <<'PY'
import torch, torchvision, phantom
assert torch.cuda.is_available(), "no CUDA device visible"
cc = torch.cuda.get_device_capability()
sm = f"sm_{cc[0]}{cc[1]}"
archs = torch.cuda.get_arch_list()
print("torch", torch.__version__, "| cuda", torch.version.cuda,
      "| torchvision", torchvision.__version__)
print("gpu  ", torch.cuda.get_device_name(0), sm)
print("wheel arch list:", archs)
assert sm in archs, (
    f"this torch wheel was NOT built for {sm} — wrong --extra-index-url? "
    f"(built for {archs})")
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
assert torch.isfinite(x @ x).all(), "bf16 matmul produced non-finite values"
print("bf16 matmul on", sm, "OK")
PY

# --- 3. model side: build -> one step of EVERY program's loss -> NFE sample
#        -> pack/unpack round trip -> checkpoint round trip
echo
echo "===== [2/6] tiny model smoke (phantom.scripts.smoke_test) ====="
"$VENV/bin/python" -m phantom.scripts.smoke_test --tiny --synthetic --device cuda

# --- 4. the real training CLI: 2 steps -> a checkpoint on disk ---------------
echo
echo "===== [3/6] train_teacher: 2 tiny steps -> checkpoint ====="
"$VENV/bin/python" -m phantom.train.train_teacher \
    --tiny --synthetic --device cuda \
    --max-steps 2 --ckpt-every 2 --run-name "gpu_smoke_${STAMP}"
TEACHER="runs/teacher/gpu_smoke_${STAMP}/teacher_000002.pt"
test -f "$TEACHER" || { echo "FAIL: no teacher checkpoint at $TEACHER" >&2; exit 1; }
echo "teacher checkpoint: $TEACHER"

# --- 5. the real distillation CLI: 2 steps, then a --resume off its own
#        checkpoint (the path that silently reverted the recipe pre-0918)
echo
echo "===== [4/6] distill_hid: 2 tiny steps -> checkpoint -> --resume ====="
"$VENV/bin/python" -m phantom.train.distill_hid \
    --tiny --synthetic --device cuda \
    --max-steps 2 --ckpt-every 2 --run-name "gpu_smoke_${STAMP}"
STUDENT="runs/hid/gpu_smoke_${STAMP}_r0/student_000002.pt"
test -f "$STUDENT" || { echo "FAIL: no student checkpoint at $STUDENT" >&2; exit 1; }
"$VENV/bin/python" -m phantom.train.distill_hid \
    --tiny --synthetic --device cuda \
    --max-steps 4 --ckpt-every 4 --resume "$STUDENT" \
    --run-name "gpu_smoke_${STAMP}_resume"
echo "student checkpoint: $STUDENT"

# --- 6. deployment: the receding-horizon loop against the MOCK rig, driven by
#        a real tiny policy on the GPU (configs/hardware.yaml is mode: mock)
echo
echo "===== [5/6] run_deploy dry run (mock rig, real tiny policy) ====="
"$VENV/bin/python" -m phantom.scripts.run_deploy \
    --system teacher --tiny --task smoke --device cuda \
    --episodes 1 --max-replans 2 --max-episode-s 20 --no-label-prompt

# --- 7. the same marked tests CI runs (they pin themselves to CPU; this
#        re-checks them against the pinned wheel + this interpreter)
echo
echo "===== [6/6] pytest ====="
if [ "${FULL:-0}" = "1" ]; then
  "$VENV/bin/python" -m pytest -q -rs -m "not slow"
else
  "$VENV/bin/python" -m pytest -q -rs -m "model_smoke and not slow"
fi

echo
echo "GPU SMOKE PASSED — venv $VENV"
