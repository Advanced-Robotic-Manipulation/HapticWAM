"""JPEG frame codec for scene/camera RGB storage in zarr episodes.

Scene RGB at 640x480x3 uint8 @ 30 fps is ~1.6 GB/min raw — the dominant
disk/offload cost. Each frame is JPEG-encoded and stored as a variable-length
byte string (zarr VLenBytes object array). The reader decodes transparently
via `JpegFrameArray`, so every consumer keeps receiving HxWx3 uint8 arrays and
the EpisodeReader API is unchanged.

Backend preference: cv2 (fast, present on the rig via the `hw` extra) ->
Pillow (portable fallback, a base dependency). Both produce/consume standard
RGB JPEG files, so an episode encoded with one backend decodes correctly with
the other, and decode(encode(x)) is identity up to JPEG's lossy quantization.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np

# cached backend: ("cv2", module) | ("pil", Image class)
_BACKEND = None


def _backend():
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    try:
        import cv2  # noqa: F401
        _BACKEND = ("cv2", cv2)
        return _BACKEND
    except Exception:
        pass
    try:
        from PIL import Image
        _BACKEND = ("pil", Image)
        return _BACKEND
    except Exception as e:  # pragma: no cover - only if both missing
        raise RuntimeError(
            "scene JPEG storage/decoding needs OpenCV (cv2) or Pillow, but "
            "neither is importable") from e


def encode_jpeg(frame: np.ndarray, quality: int = 92) -> bytes:
    """Encode an HxWx3 (or HxW) uint8 frame to JPEG bytes."""
    frame = np.ascontiguousarray(frame)
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    kind, mod = _backend()
    if kind == "cv2":
        cv2 = mod
        img = frame
        if img.ndim == 3 and img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(
            ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return buf.tobytes()
    # Pillow
    Image = mod
    im = Image.fromarray(frame)
    bio = BytesIO()
    im.save(bio, format="JPEG", quality=int(quality))
    return bio.getvalue()


def decode_jpeg(data) -> np.ndarray:
    """Decode JPEG bytes to an HxWx3 uint8 RGB array."""
    b = bytes(data)
    kind, mod = _backend()
    if kind == "cv2":
        cv2 = mod
        arr = np.frombuffer(b, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("cv2.imdecode failed (corrupt JPEG)")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # Pillow
    Image = mod
    im = Image.open(BytesIO(b))
    return np.asarray(im.convert("RGB"))


class JpegFrameArray:
    """Read-only array-like view over a zarr VLenBytes array of JPEG frames.

    Decodes lazily on access and presents a dense-array face — `shape`
    (T, H, W, C), `dtype` uint8, `ndim`, `size`, `len()`, integer and slice
    indexing — so consumers that inspect or slice a normal zarr `data` array
    keep working without knowing the frames are JPEG-encoded.
    """

    def __init__(self, z, frame_shape, dtype="uint8"):
        self._z = z
        self._frame_shape = tuple(int(x) for x in frame_shape)
        self.dtype = np.dtype(dtype)

    @property
    def shape(self) -> tuple[int, ...]:
        return (int(self._z.shape[0]), *self._frame_shape)

    @property
    def ndim(self) -> int:
        return 1 + len(self._frame_shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    def __len__(self) -> int:
        return int(self._z.shape[0])

    def _decode_one(self, b) -> np.ndarray:
        img = decode_jpeg(b).astype(self.dtype, copy=False)
        if img.shape != self._frame_shape:
            img = img.reshape(self._frame_shape)
        return img

    def _decode_many(self, raw) -> np.ndarray:
        if len(raw) == 0:
            return np.empty((0, *self._frame_shape), dtype=self.dtype)
        return np.stack([self._decode_one(b) for b in raw])

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return self._decode_one(self._z[int(key)])
        if isinstance(key, slice):
            return self._decode_many(list(self._z[key]))
        # list/array of indices, or anything else zarr understands
        raw = self._z[key]
        return self._decode_many(list(raw))

    def __array__(self, dtype=None):
        arr = self._decode_many(list(self._z[:]))
        return arr.astype(dtype) if dtype is not None else arr
