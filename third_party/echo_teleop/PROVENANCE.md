# Echo right-arm teleop — vendored from compute2 (2026-07-10)

The lab's working teleop + dataset-collection stack for the RIGHT UR3 arm,
extracted as the MINIMAL runtime closure of `main11.py` from
`compute2:/media/isr-lab-4/Main/kngn_ws/Echo-Yolo/Echo/`.

**Why vendored:** `main11.py` was UNTRACKED in that repo (git remote
`khang123452/ur_base`, 87 dirty paths) — it existed only on that disk, edited
same-day as extraction. This copy is the durable reference. Nothing was
modified; permissions/bytes as found.

## What it is

- `main11.py` — entry point: Echo leader-device teleop of the right UR3
  (`192.168.88.56`) at 100 Hz `servoJ` (lookahead 0.1, gain 200) + Robotiq
  2F-85 (URCap socket :63352, binary open 5/closed 155) + one RealSense
  RGB+aligned-depth 1280x720@30 → records episodes at 30 Hz into `dataCoRL/`
  (center-crop 720 → 244x244 rgb jpg + uint16 depth png + robot_states.npz +
  metadata.json). Recording start/stop comes from the Echo device's own flag
  (byte [4]) or the `r` key; `q` quits; 3-level sensitivity from the device.
- `echo_teleoperation.py` — the Echo leader: custom STM32 exo/glove over USB
  serial (VID 1603 / PID 1868, 115200 baud), 8x int16 joint ticks per hand
  (index 7/15 = grippers) + record flag + sensitivity; `read_pose_rad()`.
- `ur_rtde.py` — `UR3Teleop`: RTDEControl + RTDEReceive wrapper (+ gripper).
- `camera.py` — `RealSenseCamera` (RGB + aligned z16 depth).
- `robotiq_gripper.py` — vendored Robotiq socket driver (stdlib-only).

Import chain: `main11` → {echo_teleoperation, ur_rtde, camera};
`ur_rtde` → `robotiq_gripper`. No YOLO / torch / model weights / calibration
files — every constant (IPs, base pose, gripper cfg) is inline in `__main__`.

## Real deps (their requirements.txt is stale — lists torch/ultralytics
leftovers and OMITS two needed libs)

    numpy, opencv-python, pyserial, pyrealsense2, ur_rtde   (py3.12 ok)

On compute2 it runs in conda env `echo` (py3.12.11, ur_rtde 1.6.2):
`conda activate echo && python main11.py`.

## Hardware facts learned

- TWO UR3 arms: left `192.168.88.40`, right `192.168.88.56` (main11 = right
  only). Generation (CB3 vs e-Series) NOT determinable from software — read
  the pendant/plate; decides `arm.generation` + wrist F/T source in
  `configs/hardware.yaml` (CB3 ⇒ external FT-300S needed for ACC).
- Robotiq 2F-85; single RealSense; existing recordings: 169 episodes / 6.5 GB
  in `Echo/dataCoRL/` on compute2 (left in place).

## Integration notes (for the PHANTOM recorder — not done yet, by design)

1. Echo is JOINT-space teleop (servoJ with absolute joint targets); phantom's
   action schema is Δ-EE pose (6) + gripper (1). Same dim (7), different
   semantics — decide at integration: either record joint-space actions
   (config semantics change) or derive Δ-EE from RTDE TCP pose during
   recording (keeps the Cosmos action-cond convention).
2. The clean integration point is a `phantom/teleop/echo.py` TeleopDevice
   wrapping `echo_teleoperation.Echo` (device already provides the episode
   start/stop flag — maps to recorder start/stop), leaving main11.py as the
   lab's standalone reference.
3. dataCoRL episodes are convertible (rgb+depth+npz → phantom zarr streams)
   if the existing 169 episodes ever become useful; tactile obviously absent.
