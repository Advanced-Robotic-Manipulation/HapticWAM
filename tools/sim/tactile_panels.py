"""Native tactile evidence and explicitly uncalibrated comparison panels.

Recorded infer_img pixels are preserved losslessly. Proxy pixels are a simple
force-driven Gaussian, not a DM-Tac optical/contact reconstruction. This module
does not access hardware or change an existing replay export.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SIDES = ("left", "right")


def timestamps(values, name):
    values = np.asarray(values, dtype=np.float64)
    if (
        values.ndim != 1
        or len(values) < 2
        or not np.isfinite(values).all()
        or np.any(np.diff(values) <= 0)
    ):
        raise ValueError(f"{name}: need finite strictly increasing native timestamps")
    return values


def previous_sample(ts, t):
    """Causal display: never borrow a future tactile observation."""
    index = int(np.searchsorted(ts, t, side="right") - 1)
    return (index, float(t - ts[index])) if index >= 0 else (-1, None)


def export_tactile_sidecar(episode: Path, prepared: Path, out: Path):
    import cv2

    from phantom.data.episode_store import EpisodeReader

    episode, prepared, out = (p.resolve() for p in (episode, prepared, out))
    if episode == out or episode in out.parents or out == prepared:
        raise ValueError(
            "Tactile output must be separate from source and existing replay"
        )
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to replace tactile evidence: {out}")
    replay_path = prepared / "replay.npz"
    replay_bytes = replay_path.read_bytes()
    source_meta = (episode / "meta.json").read_bytes()
    parent = json.loads((prepared / "manifest.json").read_text())
    if Path(parent["episode"]).name != episode.name:
        raise ValueError("Source episode and prepared replay do not match")
    with np.load(replay_path, allow_pickle=False) as replay:
        t0 = float(replay["t0_master"])
    reader = EpisodeReader(episode)
    arrays = {"t0_master": np.asarray(t0)}
    streams = {}
    for side in SIDES:
        for kind, stream_kind in (
            ("image", "infer_img"),
            ("wrench", "wrench"),
            ("depth", "fields_ds"),
        ):
            stream = f"tactile_{side}_{stream_kind}"
            if not reader.has(stream):
                if kind == "image":
                    raise ValueError(f"Required tactile image stream missing: {stream}")
                continue
            ts = timestamps(reader.ts(stream), stream) - t0
            values = np.asarray(reader.data(stream)[:])
            if kind == "depth":
                if values.ndim != 4 or values.shape[-1] != 8:
                    raise ValueError(f"{stream}: expected canonical 8-channel field")
                values = values[..., 2]
            if kind == "image" and (
                values.dtype != np.uint8 or values.ndim not in (3, 4)
            ):
                raise ValueError(
                    f"{stream}: expected native uint8 grayscale/RGB images"
                )
            if len(values) != len(ts) or not np.isfinite(values).all():
                raise ValueError(f"{stream}: nonfinite or mismatched samples")
            arrays[f"{side}_{kind}"] = values
            arrays[f"{side}_{kind}_t"] = ts
            streams[stream] = {
                "samples": len(ts),
                "shape": list(values.shape),
                "dtype": str(values.dtype),
                "median_hz": float(1 / np.median(np.diff(ts))),
                "max_gap_s": float(np.diff(ts).max()),
                "first_t_s": float(ts[0]),
                "last_t_s": float(ts[-1]),
            }
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "tactile.npz", **arrays)
    tiles = []
    for side in SIDES:
        values, ts = arrays[f"{side}_image"], arrays[f"{side}_image_t"]
        row = []
        for index in np.linspace(0, len(values) - 1, 4).astype(int):
            image = values[index]
            bgr = cv2.cvtColor(
                image, cv2.COLOR_GRAY2BGR if image.ndim == 2 else cv2.COLOR_RGB2BGR
            )
            cv2.imwrite(str(out / f"{side}_{index:04d}.png"), bgr)
            tile = cv2.resize(bgr, (256, 192))
            cv2.putText(
                tile,
                f"REAL {side} t={ts[index]:.2f}s",
                (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )
            row.append(tile)
        tiles.append(np.concatenate(row, axis=1))
    cv2.imwrite(str(out / "tactile_contact_sheet.jpg"), np.concatenate(tiles, axis=0))
    manifest = {
        "schema_version": 1,
        "episode": str(episode),
        "split": parent["split"],
        "t0_master_s": t0,
        "reference_replay_sha256": hashlib.sha256(replay_bytes).hexdigest(),
        "reference_meta_sha256": parent["meta_sha256"],
        "source_meta_sha256": hashlib.sha256(source_meta).hexdigest(),
        "source_zarr_open_mode": "r",
        "streams": streams,
        "image_encoding": "Native uint8 infer_img pixels preserved losslessly in NPZ; no resizing or exposure adjustment",
        "depth_encoding": "Native canonical fields_ds channel2; SDK indentation units, not calibrated metres",
        "time_domain": "Native MasterClock timestamps minus existing replay t0_master; no clock offset reapplied",
        "source_and_reference_unchanged": (
            (episode / "meta.json").read_bytes() == source_meta
            and replay_path.read_bytes() == replay_bytes
        ),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "tactile_export": str(out),
                "episode": episode.name,
                "split": parent["split"],
            }
        ),
        flush=True,
    )
    return manifest


def force_proxy(force, *, uv=None, size=(320, 240)):
    """Return BGR force proxy; unknown location is explicitly a centered prior."""
    import cv2

    force = np.asarray(force, dtype=float)
    if force.shape != (3,) or not np.isfinite(force).all():
        raise ValueError("Proxy force must be finite xyz")
    center = np.zeros(2) if uv is None else np.asarray(uv, dtype=float)
    if center.shape != (2,) or not np.isfinite(center).all() or np.any(abs(center) > 1):
        raise ValueError("Contact UV must be finite normalized [-1,1] coordinates")
    width, height = size
    yy, xx = np.mgrid[-1 : 1 : complex(height), -1 : 1 : complex(width)]
    pressure = np.exp(-((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / 0.18)
    pressure *= min(float(np.linalg.norm(force)) / 15, 0.8)
    gel = np.clip(70 + 150 * pressure, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gel, cv2.COLOR_GRAY2BGR)


class TactilePanels:
    def __init__(self, directory, sim, *, max_age_s=0.5):
        self.directory = Path(directory)
        self.data = np.load(self.directory / "tactile.npz", allow_pickle=False)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        self.sim, self.max_age_s = sim, max_age_s
        self.sim_t = timestamps(sim["t"], "simulation force t")
        key = (
            "pad_packet_force"
            if "pad_packet_force" in sim
            else "pad_force"
            if "pad_force" in sim
            else "pad_forces"
        )
        if key not in sim:
            raise ValueError("Tactile comparison needs measured pad_force (N,2,3)")
        self.force = np.asarray(sim[key])
        self.force_source = key
        if (
            self.force.shape != (len(self.sim_t), 2, 3)
            or not np.isfinite(self.force).all()
        ):
            raise ValueError("pad_force must be finite (N,2,3)")
        self.uv = np.asarray(sim["pad_contact_uv"]) if "pad_contact_uv" in sim else None
        if self.uv is not None and self.uv.shape != (len(self.sim_t), 2, 2):
            raise ValueError("pad_contact_uv must be (N,2,2)")
        self.normal = (
            np.asarray(sim["pad_packet_normal_force"])
            if "pad_packet_normal_force" in sim
            else None
        )
        if self.normal is not None and (
            self.normal.shape != (len(self.sim_t), 2)
            or not np.isfinite(self.normal).all()
            or np.any(self.normal < 0)
        ):
            raise ValueError("pad_packet_normal_force must be finite nonnegative (N,2)")
        for side in SIDES:
            ts = timestamps(self.data[f"{side}_image_t"], side)
            if len(ts) != len(self.data[f"{side}_image"]):
                raise ValueError(f"{side}: tactile frame/timestamp mismatch")
        self.rows = []

    def render(self, t, *, scene_width=640, sim_time=None):
        """Keep native tactile causal at t; align proxy to displayed sim time.

        Offline comparison can display a nearest simulation frame slightly
        after the common video grid. Sampling its force at that exact frame
        time avoids accidentally showing the preceding full-frame sample.
        Callers without an explicit simulation time retain previous behavior.
        """
        import cv2

        w, h = scene_width // 2, 240
        row = np.zeros((h + 60, scene_width * 2, 3), dtype=np.uint8)
        force_time = float(t if sim_time is None else sim_time)
        if not np.isfinite(force_time):
            raise ValueError("Simulation force query time must be finite")
        fi, force_age = previous_sample(self.sim_t, force_time)
        alignment = {
            "t": float(t),
            "sim_displayed_frame_t": None if sim_time is None else force_time,
            "sim_force_query_t": force_time,
            "sim_force_sample_t": float(self.sim_t[fi]) if fi >= 0 else None,
            "sim_force_index": fi,
            "sim_force_age_s": force_age,
            "sim_force_time_offset_from_common_s": float(self.sim_t[fi] - t) if fi >= 0 else None,
        }
        for index, side in enumerate(SIDES):
            image_t = self.data[f"{side}_image_t"]
            sample, age = previous_sample(image_t, t)
            available = sample >= 0 and age <= self.max_age_s
            alignment[f"{side}_image_index"] = sample
            alignment[f"{side}_image_age_s"] = age
            alignment[f"{side}_image_fresh"] = available
            real = np.zeros((h, w, 3), np.uint8)
            if sample >= 0:
                pixels = self.data[f"{side}_image"][sample]
                pixels = cv2.cvtColor(
                    pixels,
                    cv2.COLOR_GRAY2BGR if pixels.ndim == 2 else cv2.COLOR_RGB2BGR,
                )
                real = cv2.resize(pixels, (w, h), interpolation=cv2.INTER_AREA)
            label = (
                "no sample"
                if age is None
                else f"age {age * 1000:.0f}ms" + (" STALE" if not available else "")
            )
            force = self.force[fi, index] if fi >= 0 else np.zeros(3)
            uv = self.uv[fi, index] if self.uv is not None and fi >= 0 else None
            known_uv = uv is not None and np.isfinite(uv).all()
            pressure_force = (
                np.array([0, 0, self.normal[fi, index]])
                if self.normal is not None and fi >= 0
                else force
            )
            proxy = force_proxy(
                pressure_force, uv=uv if known_uv else None, size=(w, h)
            )
            alignment[f"{side}_sim_force_norm_N"] = float(np.linalg.norm(force))
            alignment[f"{side}_pressure_force_N"] = float(
                np.linalg.norm(pressure_force)
            )
            alignment[f"{side}_contact_location"] = (
                "simulated_contact_uv" if known_uv else "centered_prior"
            )
            wrench_label = f"wrench unavailable; {label}"
            key = f"{side}_wrench_t"
            if key in self.data:
                wi, wa = previous_sample(self.data[key], t)
                if wi >= 0:
                    value = float(np.linalg.norm(self.data[f"{side}_wrench"][wi, :3]))
                    alignment[f"{side}_real_force_norm"] = value
                    alignment[f"{side}_real_force_age_s"] = wa
                    wrench_label = f"recorded |F|={value:.2f}; {label}"
            simulated_force_label = (
                f"Fn={np.linalg.norm(pressure_force):.2f} N"
                if self.normal is not None
                else f"|F|={np.linalg.norm(force):.2f} N"
            )
            for x, pixels, title, subtitle in [
                (index * w, real, f"REAL {side.upper()} / infer_img", wrench_label),
                (
                    scene_width + index * w,
                    proxy,
                    f"SIM {side.upper()} / UNCALIBRATED PROXY",
                    simulated_force_label
                    + "; "
                    + ("contact UV" if known_uv else "center assumed"),
                ),
            ]:
                row[52 : 52 + h, x : x + w] = pixels
                cv2.putText(
                    row,
                    title,
                    (x + 5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.44,
                    (240, 240, 240),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    row,
                    subtitle,
                    (x + 5, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (180, 180, 220),
                    1,
                    cv2.LINE_AA,
                )
            if not available:
                cv2.rectangle(
                    row,
                    (index * w, 52),
                    ((index + 1) * w - 1, 52 + h - 1),
                    (0, 90, 230),
                    2,
                )
            if fi < 0:
                cv2.putText(
                    row,
                    "NO SIM FORCE SAMPLE",
                    (scene_width + index * w + 5, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 90, 230),
                    1,
                )
        self.rows.append(alignment)
        return row

    def save(self, out):
        result = {
            "schema_version": 1,
            "reference": str(self.directory),
            "native_streams": self.manifest["streams"],
            "frames": len(self.rows),
            "sampling": "Previous native tactile observation, never a future frame; stale samples visibly marked",
            "sim_force_sampling": "Previous simulation force at the displayed simulation frame timestamp when supplied; otherwise at common video time. Force age is relative to this explicit query time; signed offset from common time is saved separately.",
            "stale_after_s": self.max_age_s,
            "sim_force_source": self.force_source,
            "pressure_source": "packet normal contact force"
            if self.normal is not None
            else "net force magnitude approximation",
            "fresh_image_frames": {
                side: sum(r[f"{side}_image_fresh"] for r in self.rows) for side in SIDES
            },
            "proxy": "Gaussian pressure image from measured pad-force magnitude; centered prior unless pad_contact_uv provided",
            "limitations": [
                "Not a calibrated DM-Tac optical, indentation, shear or slip simulation.",
                "Contact UV is a simulated pad-local manifold centroid, not a calibrated projection onto the gel image; assembly contacts may include backing/linkage.",
                "No pixel similarity score between measured gel imagery and synthetic pressure proxy is claimed.",
                "Recorded resultant wrench can include bias and uses its own timestamps.",
            ],
            "alignment": self.rows,
        }
        (Path(out) / "tactile_alignment.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        self.data.close()
        return {key: value for key, value in result.items() if key != "alignment"}
