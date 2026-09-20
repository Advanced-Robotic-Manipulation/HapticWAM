# Requirements — which file for which machine

| Machine | File | Role |
|---|---|---|
| RTX 5090 box (Ubuntu 22.04, wired to the UR3) | [requirements-5090.txt](requirements-5090.txt) | data collection, deployment, DAgger rollouts, eval |
| 8×H100 cloud cluster | [requirements-h100.txt](requirements-h100.txt) | all training / fine-tuning |
| 8×A100 cloud cluster | [requirements-a100.txt](requirements-a100.txt) | all training / fine-tuning (H100 alternative) |
| Windows/any dev box | `pip install -e .[dev,train]` from the repo root | code, tests, mock-mode smoke |

All Linux files pin the same PyTorch cu129 pair — `torch==2.13.0` +
`torchvision==0.28.0` (2026-09-19 bump from 2.7.1/0.22.1, which four GitHub
advisories flagged; the cu128 index stops at torch 2.11.0, and cu129 is still
CUDA 12.x so the driver requirement is unchanged). Those wheels are built for
sm_{75,80,86,90,100,120}, i.e. Blackwell's sm_120, H100's sm_90 and A100's
sm_80 alike. They also pin the phantom core, the pip-installable part of the
cosmos-predict2.5 import chain, and the test stack. They differ only in the
hardware drivers (5090) and the optional
transformer_engine note (H100/A100). After installing on a cluster, select
the training compute target with one gitignored line —
`echo "target: h100x8" > configs/compute.local.yaml` (or `a100x8`) — see
`configs/compute.yaml` and docs/training_playbook.md.

## Install on a fresh machine

```bash
git clone <this repo> && cd icra2027
# also place the cosmos repo + weights and point configs/paths.local.yaml at them:
#   cosmos_repo: /path/to/cosmos-predict2.5
#   cosmos_weights_root: /path/to/cosmos-predict2.5-2b

python -m venv ~/venvs/phantom && source ~/venvs/phantom/bin/activate
pip install -r requirements/requirements-<machine>.txt
pip install -e .
```

## Manual (non-pip) steps

**5090 box only:**
1. `dmrobotics` SDK (DM-Tac W2L) — **needs Python 3.8–3.11 and numpy<2**.
   The tactile worker processes are spawned from the phantom process and
   inherit its interpreter, so the WHOLE 5090 venv must be py3.11 with
   numpy<2 (torch is fine with that). Install the SDK into the same venv from
   the vendor `SDK_Publish_1.2.10/`: `pip install .[gpu]`, then build the
   TensorRT engines once per machine: `dmrobotics trt rebuild`.
   Verified API + gotchas: [../docs/sensor_sdk.md](../docs/sensor_sdk.md).
2. librealsense udev rules (RealSense without root).
3. `sudo apt install libhidapi-hidraw0 libhidapi-libusb0` + 3Dconnexion udev
   rule for the SpaceMouse.
4. `sudo usermod -aG dialout $USER`.
5. NVIDIA driver ≥ 570 (sm_120).

**H100 cluster only (optional):** real `transformer_engine` for the one-time
`verify_backbone --check-ref` shim-parity run — use an NGC PyTorch container
rather than pip-building it. All training runs on the pure-torch SDPA backend
regardless.

## Post-install verification (each machine)

```bash
pytest tests/
python -m phantom.scripts.mock_smoke                       # both
python -m phantom.scripts.smoke_test --synthetic --device cuda
# 5090: python -m phantom.scripts.verify_backbone --save-ref ref_shim.pt
# H100: python -m phantom.scripts.verify_backbone --check-ref ref_shim.pt  (with TE)
```

Full command reference: [../docs/launch_guide.md](../docs/launch_guide.md).
