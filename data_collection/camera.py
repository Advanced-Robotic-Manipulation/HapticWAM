import numpy as np
import cv2

try:
    import pyrealsense2 as rs
    _RS_OK = True
except Exception:
    rs = None
    _RS_OK = False


class RealSenseCamera:
    """
    RealSense color (BGR) + optional aligned depth with robust init/recovery.
    - Auto-picks a connected RS by serial (or use serial=...).
    - Color in BGR8 when possible; else RGB8 with conversion.
    - Warm-up frames to avoid “green/underexposed”.
    - Restarts pipeline automatically after repeated timeouts.
    """

    def __init__(
        self,
        capture_frequency: int = 30,
        width: int = 640,
        height: int = 480,
        warmup: int = 8,
        serial: str | None = None,
        max_timeouts_before_restart: int = 3,
        wait_timeout_ms: int = 2000,
    ):
        if not _RS_OK:
            raise RuntimeError("pyrealsense2 not available.")

        self.w = int(width)
        self.h = int(height)
        self.fps = int(capture_frequency)
        self._returns_bgr = True
        self._align = None
        self._serial = serial
        self._timeout_budget = 0
        self._timeout_limit = int(max_timeouts_before_restart)
        self._wait_ms = int(wait_timeout_ms)

        self.pipeline = None
        self.config = None
        self.profile = None

        self._start_pipeline(warmup=warmup)

    # ---------- device selection ----------
    def _pick_serial(self) -> str:
        ctx = rs.context()
        devs = ctx.query_devices()
        if len(devs) == 0:
            raise RuntimeError("No RealSense device found.")
        if self._serial:
            # verify exists
            for d in devs:
                if d.get_info(rs.camera_info.serial_number) == self._serial:
                    return self._serial
            raise RuntimeError(f"Requested RealSense serial {self._serial} not found.")
        # pick first with RGB sensor
        for d in devs:
            sn = d.get_info(rs.camera_info.serial_number)
            has_rgb = any("RGB" in s.get_info(rs.camera_info.name) for s in d.sensors)
            if has_rgb:
                return sn
        # fallback: first device
        return devs[0].get_info(rs.camera_info.serial_number)

    # ---------- pipeline start/restart ----------
    def _start_pipeline(self, warmup: int = 8):
        serial = self._pick_serial()

        # Clean previous
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(serial)

        # Color in BGR8 if possible, else RGB8
        try:
            self.config.enable_stream(rs.stream.color, self.w, self.h, rs.format.bgr8, self.fps)
            self._returns_bgr = True
        except Exception:
            self.config.enable_stream(rs.stream.color, self.w, self.h, rs.format.rgb8, self.fps)
            self._returns_bgr = False

        # Depth enabled so we can return aligned depth when requested
        self.config.enable_stream(rs.stream.depth, self.w, self.h, rs.format.z16, self.fps)

        self.profile = self.pipeline.start(self.config)

        # Align depth->color
        self._align = rs.align(rs.stream.color)

        # Try to ensure AE/AWB on color sensor
        try:
            color_sensor = None
            for s in self.profile.get_device().sensors:
                if s.get_info(rs.camera_info.name).lower().startswith("rgb"):
                    color_sensor = s
                    break
            if color_sensor:
                if color_sensor.supports(rs.option.enable_auto_exposure):
                    color_sensor.set_option(rs.option.enable_auto_exposure, 1)
                if color_sensor.supports(rs.option.enable_auto_white_balance):
                    color_sensor.set_option(rs.option.enable_auto_white_balance, 1)
        except Exception:
            pass

        # Warm-up frames
        for _ in range(max(0, int(warmup))):
            try:
                self.pipeline.wait_for_frames(1000)
            except Exception:
                break

        # reset timeout budget after a successful start
        self._timeout_budget = 0

    def _restart_pipeline(self):
        try:
            dev = self.profile.get_device() if self.profile else None
            if dev and hasattr(dev, "hardware_reset"):
                # soft reset the camera (optional; comment if not desired)
                dev.hardware_reset()
                # small pause to allow re-enumeration
                import time as _t
                _t.sleep(1.5)
        except Exception:
            pass
        self._start_pipeline(warmup=4)

    # ---------- public API ----------
    def get_frame(self, depth: bool = False):
        try:
            frames = self.pipeline.wait_for_frames(self._wait_ms)
        except Exception as e:
            # count timeouts and attempt recovery
            self._timeout_budget += 1
            if self._timeout_budget >= self._timeout_limit:
                self._restart_pipeline()
            raise RuntimeError(f"RealSense wait_for_frames failed: {e}")

        if depth:
            frames = self._align.process(frames)

        color_frame = frames.get_color_frame()
        if not color_frame:
            self._timeout_budget += 1
            if self._timeout_budget >= self._timeout_limit:
                self._restart_pipeline()
            raise RuntimeError("No RealSense color frame")

        color_img = np.asanyarray(color_frame.get_data())
        if not self._returns_bgr:
            color_img = color_img[:, :, ::-1].copy()

        if not depth:
            self._timeout_budget = 0
            return color_img

        depth_frame = frames.get_depth_frame()
        depth_img = None
        if depth_frame:
            depth_img = np.asanyarray(depth_frame.get_data())
            if depth_img.ndim == 2:
                depth_img = depth_img[..., np.newaxis]

        self._timeout_budget = 0
        return color_img, depth_img

    def recover(self):
        """Force a restart of the pipeline (call from the app on persistent errors)."""
        self._restart_pipeline()

    def release(self):
        try:
            if self.pipeline:
                self.pipeline.stop()
        except Exception:
            pass


class WebCamera:
    def __init__(self, camera_id, width: int = 1280, height: int = 720, fps: int = 30, prefer_mjpg: bool = True):
        api = cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else cv2.CAP_ANY
        self.camera = cv2.VideoCapture(camera_id, api)
        if not self.camera.isOpened():
            self.camera.release()
            raise RuntimeError(f"Cannot open camera index {camera_id}")

        if prefer_mjpg:
            try:
                self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            except Exception:
                pass
        self.camera.set(cv2.CAP_PROP_FPS, fps)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        ok, frame = self.camera.read()
        if not ok or frame is None:
            self.camera.release()
            raise RuntimeError(f"Cannot read from camera index {camera_id}")

    def get_frame(self):
        ok, image = self.camera.read()
        if not ok or image is None:
            raise RuntimeError("Cannot read from the webcam")
        return image

    def release(self):
        self.camera.release()
