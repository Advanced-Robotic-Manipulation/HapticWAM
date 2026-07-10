import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import json
import time
import cv2
import numpy as np

from echo_teleoperation import Echo 
from ur_rtde import UR3Teleop
from camera import RealSenseCamera


def check_opencv_gui():
    try:
        win = "test_gui"
        cv2.namedWindow(win)
        cv2.destroyWindow(win)
        return True
    except cv2.error:
        return False


def provide_teleoperation_right_arm_only(
    right_arm,
    right_base_pose,
    data_from_echo,
    binary_gripper_pose,
    gripper_config,
):
    if data_from_echo is None or right_arm is None:
        return

    right_arm_position = data_from_echo[1]
    right_gripper_position = data_from_echo[3]

    if binary_gripper_pose:
        right_gripper_position = (
            gripper_config["gripper_closed_pose"]
            if right_gripper_position >= gripper_config["gripper_pose_threshold"]
            else gripper_config["gripper_opened_pose"]
        )

    sensitivity_mode = data_from_echo[5] if len(data_from_echo) > 5 else 0
    if sensitivity_mode == 0:
        right_arm_new_position = right_base_pose + right_arm_position
    elif sensitivity_mode == 1:
        right_arm_new_position = right_base_pose + right_arm_position / 1.25
    else:
        right_arm_new_position = right_base_pose + right_arm_position / 1.75

    right_arm.move_to_pose(
        joints_positions=right_arm_new_position,
        gripper_position=right_gripper_position,
    )


def center_crop_square(frame, crop_size):
    if frame is None:
        return None

    h, w = frame.shape[:2]
    crop_size = min(int(crop_size), h, w)
    start_x = (w - crop_size) // 2
    start_y = (h - crop_size) // 2
    return frame[start_y:start_y + crop_size, start_x:start_x + crop_size]


def crop_and_resize_rgb(frame, crop_size, output_size):
    cropped = center_crop_square(frame, crop_size)
    if cropped is None:
        return None
    return cv2.resize(cropped, output_size, interpolation=cv2.INTER_AREA)


def crop_and_resize_depth_for_display(depth_frame, crop_size, output_size):
    if depth_frame is None:
        return None

    if depth_frame.ndim == 3:
        depth_frame = depth_frame[:, :, 0]

    depth_cropped = center_crop_square(depth_frame, crop_size)
    depth_resized = cv2.resize(depth_cropped, output_size, interpolation=cv2.INTER_NEAREST)
    valid_depth = depth_resized[depth_resized > 0]
    if valid_depth.size == 0:
        depth_8u = np.zeros_like(depth_resized, dtype=np.uint8)
    else:
        near = np.percentile(valid_depth, 2)
        far = np.percentile(valid_depth, 98)
        if far <= near:
            far = near + 1.0
        depth_clipped = np.clip(depth_resized, near, far)
        depth_8u = ((depth_clipped - near) * 255.0 / (far - near)).astype(np.uint8)

    return cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)


def crop_and_resize_depth_raw(depth_frame, crop_size, output_size):
    if depth_frame is None:
        return None

    if depth_frame.ndim == 3:
        depth_frame = depth_frame[:, :, 0]

    depth_cropped = center_crop_square(depth_frame, crop_size)
    return cv2.resize(depth_cropped, output_size, interpolation=cv2.INTER_NEAREST)


def depth_stats(depth_frame):
    if depth_frame is None:
        return {
            "shape": None,
            "dtype": None,
            "min": None,
            "max": None,
            "nonzero": 0,
            "valid_min": None,
            "valid_max": None,
        }

    if depth_frame.ndim == 3:
        depth_frame = depth_frame[:, :, 0]

    valid = depth_frame[depth_frame > 0]
    return {
        "shape": list(depth_frame.shape),
        "dtype": str(depth_frame.dtype),
        "min": int(depth_frame.min()) if depth_frame.size else None,
        "max": int(depth_frame.max()) if depth_frame.size else None,
        "nonzero": int(valid.size),
        "valid_min": int(valid.min()) if valid.size else None,
        "valid_max": int(valid.max()) if valid.size else None,
    }


def colorize_depth_for_save(depth_frame):
    stats = depth_stats(depth_frame)
    if depth_frame is None:
        return None

    if depth_frame.ndim == 3:
        depth_frame = depth_frame[:, :, 0]

    if stats["nonzero"] == 0:
        depth_8u = np.zeros_like(depth_frame, dtype=np.uint8)
    else:
        near = stats["valid_min"]
        far = stats["valid_max"]
        if far <= near:
            far = near + 1
        clipped = np.clip(depth_frame, near, far)
        depth_8u = ((clipped - near) * 255.0 / (far - near)).astype(np.uint8)

    return cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)


def next_episode_number(dataset_dir):
    if not os.path.exists(dataset_dir):
        return 1

    episode_ids = []
    for name in os.listdir(dataset_dir):
        if not name.startswith("episode_"):
            continue
        path = os.path.join(dataset_dir, name)
        if not os.path.isdir(path):
            continue
        try:
            episode_ids.append(int(name.split("_")[1]))
        except (IndexError, ValueError):
            pass

    return max(episode_ids) + 1 if episode_ids else 1


def make_episode_state(dataset_dir):
    return {
        "active": False,
        "id": next_episode_number(dataset_dir),
        "frames": 0,
        "timestamps": [],
        "joint_angles": [],
        "joint_velocities": [],
        "end_effector_poses": [],
        "gripper_positions": [],
        "rgb_paths": [],
        "depth_paths": [],
        "depth_vis_paths": [],
        "depth_stats": [],
    }


def start_episode(ep, dataset_dir):
    ep["active"] = True
    ep["id"] = next_episode_number(dataset_dir)
    ep["frames"] = 0
    for key in (
        "timestamps",
        "joint_angles",
        "joint_velocities",
        "end_effector_poses",
        "gripper_positions",
        "rgb_paths",
        "depth_paths",
        "depth_vis_paths",
        "depth_stats",
    ):
        ep[key] = []

    episode_dir = os.path.join(dataset_dir, f"episode_{ep['id']:06d}")
    os.makedirs(os.path.join(episode_dir, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(episode_dir, "depth"), exist_ok=True)
    os.makedirs(os.path.join(episode_dir, "depth_vis"), exist_ok=True)
    os.makedirs(os.path.join(episode_dir, "data"), exist_ok=True)
    print(f"[dataCoRL] START episode_{ep['id']:06d}")


def read_right_arm_state(right_arm):
    joint_angles = np.zeros(6, dtype=np.float32)
    joint_velocities = np.zeros(6, dtype=np.float32)
    end_effector_pose = np.zeros(6, dtype=np.float32)
    gripper_position = np.array([0.0], dtype=np.float32)

    if right_arm is None:
        return joint_angles, joint_velocities, end_effector_pose, gripper_position

    try:
        joints = right_arm.get_current_joint_angles()
        if joints is not None:
            joint_angles[:] = np.asarray(joints, dtype=np.float32)[:6]
    except Exception as e:
        print(f"[dataCoRL] joint angle read error: {e}")

    try:
        if hasattr(right_arm, "rtde_r") and hasattr(right_arm.rtde_r, "getActualQd"):
            velocities = right_arm.rtde_r.getActualQd()
            if velocities is not None:
                joint_velocities[:] = np.asarray(velocities, dtype=np.float32)[:6]
    except Exception as e:
        print(f"[dataCoRL] joint velocity read error: {e}")

    try:
        pose = right_arm.get_current_tcp_pose()
        if pose is not None:
            end_effector_pose[:] = np.asarray(pose, dtype=np.float32)[:6]
    except Exception as e:
        print(f"[dataCoRL] end-effector pose read error: {e}")

    try:
        if hasattr(right_arm, "get_current_gripper_pose"):
            gripper = right_arm.get_current_gripper_pose()
            if gripper is not None:
                gripper_position[:] = np.asarray(gripper, dtype=np.float32).reshape(-1)[:1]
    except Exception as e:
        print(f"[dataCoRL] gripper read error: {e}")

    return joint_angles, joint_velocities, end_effector_pose, gripper_position


def record_frame(ep, dataset_dir, rgb_frame, depth_frame, right_arm, crop_size, output_size):
    if not ep["active"] or rgb_frame is None or depth_frame is None:
        return

    episode_name = f"episode_{ep['id']:06d}"
    episode_dir = os.path.join(dataset_dir, episode_name)
    rgb_dir = os.path.join(episode_dir, "rgb")
    depth_dir = os.path.join(episode_dir, "depth")
    depth_vis_dir = os.path.join(episode_dir, "depth_vis")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(depth_vis_dir, exist_ok=True)

    frame_id = ep["frames"]
    rgb_small = crop_and_resize_rgb(rgb_frame, crop_size, output_size)
    depth_raw = crop_and_resize_depth_raw(depth_frame, crop_size, output_size)

    if rgb_small is None or depth_raw is None:
        print(f"[dataCoRL] frame {frame_id:06d}: skipped because rgb/depth is None")
        return

    current_depth_stats = depth_stats(depth_raw)
    if current_depth_stats["nonzero"] == 0:
        print(f"[dataCoRL] WARNING frame {frame_id:06d}: depth has no nonzero pixels")

    rgb_path = os.path.join(rgb_dir, f"frame_{frame_id:06d}.jpg")
    depth_path = os.path.join(depth_dir, f"frame_{frame_id:06d}.png")
    depth_vis_path = os.path.join(depth_vis_dir, f"frame_{frame_id:06d}.jpg")

    rgb_ok = cv2.imwrite(rgb_path, rgb_small)
    depth_ok = cv2.imwrite(depth_path, depth_raw)
    depth_vis = colorize_depth_for_save(depth_raw)
    depth_vis_ok = cv2.imwrite(depth_vis_path, depth_vis) if depth_vis is not None else False

    if not rgb_ok or not depth_ok or not depth_vis_ok:
        print(
            f"[dataCoRL] WARNING frame {frame_id:06d}: write status "
            f"rgb={rgb_ok} depth={depth_ok} depth_vis={depth_vis_ok}"
        )

    depth_readback = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    readback_stats = depth_stats(depth_readback)
    if depth_readback is None or readback_stats["nonzero"] == 0:
        print(f"[dataCoRL] WARNING frame {frame_id:06d}: saved depth readback is empty")

    joint_angles, joint_velocities, end_effector_pose, gripper_position = read_right_arm_state(right_arm)

    ep["timestamps"].append(time.time())
    ep["joint_angles"].append(joint_angles)
    ep["joint_velocities"].append(joint_velocities)
    ep["end_effector_poses"].append(end_effector_pose)
    ep["gripper_positions"].append(gripper_position)
    ep["rgb_paths"].append(os.path.relpath(rgb_path, episode_dir))
    ep["depth_paths"].append(os.path.relpath(depth_path, episode_dir))
    ep["depth_vis_paths"].append(os.path.relpath(depth_vis_path, episode_dir))
    ep["depth_stats"].append(current_depth_stats)
    ep["frames"] += 1

    if frame_id < 3 or ep["frames"] % 30 == 0:
        print(
            f"[dataCoRL] frame {frame_id:06d}: "
            f"rgb={rgb_small.shape}/{rgb_small.dtype} saved={rgb_ok}; "
            f"depth={current_depth_stats['shape']}/{current_depth_stats['dtype']} "
            f"min={current_depth_stats['min']} max={current_depth_stats['max']} "
            f"nonzero={current_depth_stats['nonzero']} saved={depth_ok} "
            f"readback_nonzero={readback_stats['nonzero']}; "
            f"q={np.round(joint_angles, 4).tolist()} "
            f"qd={np.round(joint_velocities, 4).tolist()} "
            f"tcp={np.round(end_effector_pose, 4).tolist()}"
        )


def stop_and_save_episode(ep, dataset_dir, crop_size, output_size):
    if not ep["active"]:
        return

    ep["active"] = False
    episode_name = f"episode_{ep['id']:06d}"
    episode_dir = os.path.join(dataset_dir, episode_name)
    data_dir = os.path.join(episode_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    if ep["frames"] == 0:
        print(f"[dataCoRL] STOP {episode_name}: no frames recorded")
        return

    arrays_path = os.path.join(data_dir, "robot_states.npz")
    np.savez_compressed(
        arrays_path,
        timestamps=np.asarray(ep["timestamps"], dtype=np.float64),
        joint_angles=np.asarray(ep["joint_angles"], dtype=np.float32),
        joint_velocities=np.asarray(ep["joint_velocities"], dtype=np.float32),
        end_effector_poses=np.asarray(ep["end_effector_poses"], dtype=np.float32),
        gripper_positions=np.asarray(ep["gripper_positions"], dtype=np.float32),
        rgb_paths=np.asarray(ep["rgb_paths"], dtype=object),
        depth_paths=np.asarray(ep["depth_paths"], dtype=object),
        depth_vis_paths=np.asarray(ep["depth_vis_paths"], dtype=object),
    )

    metadata = {
        "episode_id": ep["id"],
        "frames": ep["frames"],
        "crop_size": [crop_size, crop_size],
        "output_size": [output_size[0], output_size[1]],
        "rgb_dir": "rgb",
        "depth_dir": "depth",
        "depth_vis_dir": "depth_vis",
        "state_file": os.path.relpath(arrays_path, episode_dir),
        "state_shapes": {
            "joint_angles": [ep["frames"], 6],
            "joint_velocities": [ep["frames"], 6],
            "end_effector_poses": [ep["frames"], 6],
            "gripper_positions": [ep["frames"], 1],
        },
        "depth_format": "uint16 PNG after center crop and resize",
        "depth_vis_format": "colorized JPG preview for quick visual inspection",
        "rgb_format": "BGR JPG after center crop and resize",
        "depth_stats": ep["depth_stats"],
    }
    with open(os.path.join(episode_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[dataCoRL] STOP {episode_name}: saved {ep['frames']} frames to {episode_dir}")


if __name__ == "__main__":
    GUI_AVAILABLE = check_opencv_gui()
    DATASET_DIR = "dataCoRL"
    CROP_SIZE = 720
    OUTPUT_SIZE = (244, 244)
    MAIN_W, MAIN_H = 1280, 720
    FPS = 30
    os.makedirs(DATASET_DIR, exist_ok=True)

    right_ip = "192.168.88.56"
    right_base_pose = np.array([1.427e-03, -1.5621, 1.5880, 0.00954, 1.5737, -3.0957])
    binary_gripper_pose = True
    gripper_config = {
        "gripper_opened_pose": 5,
        "gripper_closed_pose": 155,
        "gripper_pose_threshold": 150,
    }

    print("Initializing Echo device...")
    device = None
    try:
        device = Echo()
        test_data = device.read_pose_rad(dof_count=7, read_force_sensor=False)
        if test_data is not None:
            print("Echo device connected successfully.")
        else:
            print("Warning: Echo device connected but returned no data.")
    except Exception as e:
        print(f"Warning: Echo initialization failed: {e}")
        print("Camera stream will continue, but right-arm teleoperation is disabled.")

    print("Initializing right arm robot...")
    right_arm = None
    try:
        right_arm = UR3Teleop(
            ip=right_ip,
            base_pose=right_base_pose,
            lookahead_time=0.1,
            gain=200,
            binary_gripper_pose=binary_gripper_pose,
            gripper_config=gripper_config,
            use_gripper=True,
        )
        print("Right arm initialized successfully.")
    except Exception as e:
        print(f"Warning: right arm initialization failed: {e}")
        print("Camera stream will continue, but right-arm teleoperation is disabled.")

    print("Initializing RealSense RGB + depth stream...")
    main_camera = None
    try:
        main_camera = RealSenseCamera(width=MAIN_W, height=MAIN_H, capture_frequency=FPS)
        print("RealSense initialized successfully.")
    except Exception as e:
        print(f"Error: RealSense initialization failed: {e}")
        raise SystemExit(1)

    print("Starting right-arm teleoperation and RealSense stream.")
    print(f"Recording output: {DATASET_DIR}/")
    print("Recording starts/stops from Echo start_collection; press 'r' to toggle manually.")
    if GUI_AVAILABLE:
        print("Press 'q' to quit.")
    else:
        print("No OpenCV GUI available. Press Ctrl+C to stop.")

    TELEOP_UPDATE_RATE = 100
    CAMERA_UPDATE_RATE = 30
    teleop_interval = 1.0 / TELEOP_UPDATE_RATE
    camera_interval = 1.0 / CAMERA_UPDATE_RATE
    last_teleop_time = 0.0
    last_camera_time = 0.0
    ep = make_episode_state(DATASET_DIR)
    previous_start_flag = False

    try:
        while True:
            now = time.time()

            if device is not None and right_arm is not None and now - last_teleop_time >= teleop_interval:
                last_teleop_time = now
                try:
                    data_from_echo = device.read_pose_rad(dof_count=7, read_force_sensor=False)
                    if data_from_echo is not None and len(data_from_echo) > 4:
                        start_flag = bool(data_from_echo[4])
                        if start_flag and not previous_start_flag and not ep["active"]:
                            start_episode(ep, DATASET_DIR)
                        elif not start_flag and previous_start_flag and ep["active"]:
                            stop_and_save_episode(ep, DATASET_DIR, CROP_SIZE, OUTPUT_SIZE)
                        previous_start_flag = start_flag

                    provide_teleoperation_right_arm_only(
                        right_arm,
                        right_base_pose,
                        data_from_echo,
                        binary_gripper_pose,
                        gripper_config,
                    )
                except Exception as e:
                    print(f"Teleoperation read/move error: {e}")

            if now - last_camera_time >= camera_interval:
                last_camera_time = now
                try:
                    rgb_frame, depth_frame = main_camera.get_frame(depth=True)
                except Exception as e:
                    print(f"RealSense read error: {e}")
                    try:
                        main_camera.recover()
                        print("RealSense pipeline restarted.")
                    except Exception as recover_error:
                        print(f"RealSense recovery failed: {recover_error}")
                    continue

                rgb_small = crop_and_resize_rgb(rgb_frame, CROP_SIZE, OUTPUT_SIZE)
                depth_small = crop_and_resize_depth_for_display(depth_frame, CROP_SIZE, OUTPUT_SIZE)
                record_frame(ep, DATASET_DIR, rgb_frame, depth_frame, right_arm, CROP_SIZE, OUTPUT_SIZE)

                if GUI_AVAILABLE:
                    if rgb_small is not None:
                        if ep["active"]:
                            cv2.putText(
                                rgb_small,
                                f"REC {ep['id']:06d} {ep['frames']}",
                                (8, 22),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 255),
                                1,
                            )
                        cv2.imshow("RealSense RGB 244x244", rgb_small)
                    if depth_small is not None:
                        cv2.imshow("RealSense Depth 244x244", depth_small)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        print("Stopped by user.")
                        break
                    if key == ord("r"):
                        if ep["active"]:
                            stop_and_save_episode(ep, DATASET_DIR, CROP_SIZE, OUTPUT_SIZE)
                            previous_start_flag = False
                        else:
                            start_episode(ep, DATASET_DIR)

            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        if ep["active"]:
            stop_and_save_episode(ep, DATASET_DIR, CROP_SIZE, OUTPUT_SIZE)
        try:
            if main_camera is not None:
                main_camera.release()
                print("Released RealSense camera.")
        except Exception as e:
            print(f"RealSense release error: {e}")
        cv2.destroyAllWindows()
        print("main11.py ended.")
