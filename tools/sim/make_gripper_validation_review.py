#!/usr/bin/env python3
"""Annotate an existing compare_replay SBS; no Isaac, hardware or source edits.

Requires numpy, OpenCV and ffmpeg. Input video/alignment should be the untouched
outputs of the same compare_replay invocation. Contact samples are causal at the
comparison time grid, matching the existing tactile panel's sampling convention.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def times(value, name):
    value = np.asarray(value, dtype=float)
    if value.ndim != 1 or len(value) < 2 or not np.isfinite(value).all() or np.any(np.diff(value) <= 0):
        raise ValueError(f"{name} needs at least two finite increasing timestamps")
    return value


def causal_indices(ts, target, max_age):
    index = np.searchsorted(ts, target, side="right") - 1
    age = np.where(index >= 0, target - ts[index.clip(0)], np.inf)
    fresh = (index >= 0) & (age <= max_age)
    return index, age, fresh


def contact_arrays(rows):
    """Never substitute missing schema fields with normal teacher-style zeros."""
    ts = times([r["t"] for r in rows], "gel contact trace")
    active_all = np.asarray([r["normal_force_n"] for r in rows], dtype=float)
    packet = np.full((len(rows), 2), np.nan)
    ignored = np.full_like(packet, np.nan)
    coverage = set()
    for i, row in enumerate(rows):
        if len(row.get("per_pad", [])) != 2:
            raise ValueError("Need two explicitly described per_pad diagnostics")
        for side, pad in enumerate(row["per_pad"]):
            coverage.add(pad.get("coverage_mode", "unspecified"))
            ignored[i, side] = pad.get("ignored_contact_normal_force_n", np.nan)
            filters = pad.get("per_filter_contacts")
            declared = row.get("filter_paths", [[], []])[side]
            if not isinstance(filters, list) or not pad.get("filter_labels_match_columns", False):
                continue
            if "/World/Waffle" not in declared:
                continue
            entries = [f for f in filters if f.get("filter_path") == "/World/Waffle"]
            # Empty records mean no populated contacts only after explicit
            # filter-path/schema confirmation. Otherwise availability is NaN.
            if not entries:
                if filters or (pad.get("populated_unique_contact_count") == 0 and
                               pad.get("observed_filtered_normal_force_n") == 0):
                    packet[i, side] = 0.
            elif all("gel_compression_n" in f for f in entries):
                packet[i, side] = sum(float(f["gel_compression_n"]) for f in entries)
    if active_all.shape != (len(ts), 2) or not np.isfinite(active_all).all() or (active_all < 0).any():
        raise ValueError("Active gel normal_force_n must be finite nonnegative (N,2)")
    if np.any(packet[np.isfinite(packet)] < 0):
        raise ValueError("Negative active packet compression")
    return ts, active_all, packet, ignored, sorted(coverage)


def label(image, text, position, *, size=.58, color=(235, 235, 235), max_width=None):
    import cv2
    if max_width:
        width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, 1)[0][0]
        size *= min(1., max_width / max(width, 1))
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def pair_text(values, fresh):
    if not fresh:
        return "unavailable / stale"
    return "  ".join(f"{side} {value:7.3f}" if np.isfinite(value) else f"{side} unavailable"
                     for side, value in zip(["L", "R"], values)) + " N"


def run(args):
    import cv2
    inputs = {"sbs_video": args.video.resolve(), "video_alignment": args.alignment.resolve(),
              "sim_trace": args.run_dir.resolve() / "sim_trace.npz",
              "gel_trace": args.run_dir.resolve() / "gel_contact_trace.json",
              "run_metadata": args.run_dir.resolve() / "run.json"}
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Use a new output directory: {args.out}")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required for browser-compatible H.264 output")
    hashes = {k: sha(p) for k, p in inputs.items()}
    metadata = json.loads(inputs["run_metadata"].read_text())
    articulated = metadata.get("gripper_model") == "robotiq_2f85_w2l_articulated_v1"
    mode = metadata.get("mode", "unspecified")
    scope = ("SYNTHETIC PROBE; not matched real motion" if mode == "contact_probe" else
             "MEASURED-MOTION REPLAY; no policy success claim" if mode in ["replay", "dynamics"] else
             f"{mode.upper()}; diagnostic simulation")
    with np.load(inputs["sim_trace"], allow_pickle=False) as data:
        sim_t = times(data["t"], "sim t")
        frame_t = times(data["frame_t"], "sim frame_t")
        body = np.asarray(data["pad_packet_normal_force"], dtype=float)
    if body.shape != (len(sim_t), 2) or not np.isfinite(body).all() or (body < 0).any():
        raise ValueError("Need finite nonnegative pad_packet_normal_force (N,2); no net-force substitution")
    with np.load(inputs["video_alignment"], allow_pickle=False) as data:
        grid = times(data["t"], "comparison grid")
        sim_frame_t = np.asarray(data["sim_frame_t"], dtype=float)
        sim_index = np.asarray(data["sim_index"], dtype=int)
    if sim_index.shape != grid.shape or sim_frame_t.shape != grid.shape or np.any(sim_index < 0) or np.any(sim_index >= len(frame_t)):
        raise ValueError("Alignment must contain matching sim_index/sim_frame_t arrays")
    if not np.allclose(sim_frame_t, frame_t[sim_index], rtol=0, atol=1e-9):
        raise ValueError("Alignment simulation frames disagree with this run")
    step = float(np.median(np.diff(grid)))
    if not np.allclose(np.diff(grid), step, rtol=0, atol=1e-8):
        raise ValueError("Comparison video grid is not uniform")
    gel_t, active_all, active_packet, ignored, coverage = contact_arrays(json.loads(inputs["gel_trace"].read_text()))
    gi, ga, gf = causal_indices(gel_t, grid, args.max_age_s)
    bi, ba, bf = causal_indices(sim_t, grid, args.max_age_s)
    peak_scores = np.where(gf, np.nansum(active_packet[gi.clip(0)], axis=1), -1.)
    peak_frame = int(peak_scores.argmax())
    capture = cv2.VideoCapture(str(inputs["sbs_video"]))
    if not capture.isOpened():
        raise ValueError("Cannot decode SBS input")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = (int(capture.get(k)) for k in [cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT])
    if count != len(grid) or abs(fps - 1/step) > 1e-3:
        capture.release()
        raise ValueError("Input video's frame count/fps disagree with alignment")
    if width < 640 or width % 2 or height % 2:
        capture.release()
        raise ValueError("Need an even-sized SBS at least 640 pixels wide")
    args.out.mkdir(parents=True, exist_ok=True)
    top, bottom = 64, 224
    destination = args.out / "gripper_review.mp4"
    partial = args.out / "gripper_review.partial.mp4"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{width}x{height+top+bottom}", "-r", f"{1/step:.12g}", "-i", "pipe:0", "-an",
               "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(partial)]
    process = None
    try:
        with (args.out / "encode.log").open("wb") as log:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
            for i, t in enumerate(grid):
                ok, source = capture.read()
                if not ok:
                    raise ValueError(f"Input decode failed at frame {i}")
                panel = np.full((height + top + bottom, width, 3), (22, 24, 27), np.uint8)
                panel[top:top+height] = source
                label(panel, "W2L GRIPPER VALIDATION | PROVISIONAL MOUNT", (14, 25), color=(90, 200, 255), max_width=width-28)
                label(panel, f"{args.run_dir.name} | {scope}", (14, 49), size=.48, max_width=width-28)
                y = top + height
                label(panel, "SIM NORMAL CONTACT: active-window estimate versus whole body", (14, y+25), color=(90, 200, 255), max_width=width-28)
                body_name = "Whole wear-layer body -> waffle" if articulated else "Whole pad actor -> waffle (may include backing)"
                rows = [("Active gel -> waffle", active_packet[gi[i]] if gi[i] >= 0 else [np.nan]*2, gf[i], (160, 235, 150)),
                        (body_name, body[bi[i]] if bi[i] >= 0 else [np.nan]*2, bf[i], (80, 185, 255)),
                        ("Active gel, all contacts", active_all[gi[i]] if gi[i] >= 0 else [np.nan]*2, gf[i], (220, 220, 220))]
                value_x = int(width * .57)
                for j, (text, values, fresh, color) in enumerate(rows):
                    label(panel, text, (14, y+53+27*j), size=.52, max_width=value_x-28)
                    label(panel, pair_text(values, fresh), (value_x, y+53+27*j), size=.56, color=color, max_width=width-value_x-14)
                age_text = f"gel age {ga[i]*1000:.1f} ms" if gi[i] >= 0 else "gel has no prior sample"
                body_age = f"body age {ba[i]*1000:.1f} ms" if bi[i] >= 0 else "body has no prior sample"
                label(panel, f"t={t:.3f}s | {age_text}; {body_age}; sim image snap {(sim_frame_t[i]-t)*1000:+.1f} ms", (14, y+141), size=.45, max_width=width-28)
                label(panel, "Any bright SIM tactile patches use WHOLE-BODY force; they are not validated optical gel images.", (14, y+167), size=.47, color=(90, 200, 255), max_width=width-28)
                assumption = ("centred ellipse / uniform manifold pressure" if "manifold_patch_v2" in coverage else "declared uncalibrated contact filter")
                label(panel, f"Active force uses {assumption}; physical mount, shear and optical map unverified.", (14, y+191), size=.43, max_width=width-28)
                label(panel, "Contact samples use the previous observation on the comparison time grid. Missing/stale values are never zero-filled.", (14, y+213), size=.4, max_width=width-28)
                process.stdin.write(panel.tobytes())
                if i == 0:
                    cv2.imwrite(str(args.out / "review_first.png"), panel)
                if i == peak_frame:
                    cv2.imwrite(str(args.out / "review_contact_peak.png"), panel)
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError("ffmpeg failed; see encode.log")
        partial.replace(destination)
    finally:
        capture.release()
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
    aligned = {"t": grid, "sim_frame_t": sim_frame_t, "gel_index": gi, "gel_age_s": ga,
               "gel_fresh": gf, "body_index": bi, "body_age_s": ba, "body_fresh": bf,
               "active_packet_n": np.where(gf[:, None], active_packet[gi.clip(0)], np.nan),
               "active_all_n": np.where(gf[:, None], active_all[gi.clip(0)], np.nan),
               "body_packet_normal_n": np.where(bf[:, None], body[bi.clip(0)], np.nan)}
    np.savez_compressed(args.out / "review_alignment.npz", **aligned)
    assert hashes == {k: sha(p) for k, p in inputs.items()}, "Input changed during review generation"
    manifest = {"schema_version": 1, "inputs": {k: {"path": str(p), "sha256": hashes[k]} for k,p in inputs.items()},
                "helper_sha256": sha(Path(__file__)), "frames": len(grid), "fps": 1/step,
                "output_dimensions": [width, height+top+bottom], "max_age_s": args.max_age_s,
                "source_video_pixels": "All source pixels retained spatially; entire composite re-encoded to H.264",
                "sampling": "Causal previous sample at alignment.t, same grid as tactile_panels; no future sample or zero fill",
                "force_fields": {"active_waffle": "gel_contact_trace.per_pad.per_filter_contacts[/World/Waffle].gel_compression_n",
                                 "active_all": "gel_contact_trace.normal_force_n (all observed contact filters)",
                                 "whole_body_waffle": "sim_trace.pad_packet_normal_force (Gaussian tactile panel source)"},
                "whole_body_scope": "separate supplier wear-layer body; hard housings/linkage excluded" if articulated else "legacy whole-pad actor; backing/linkage may be included",
                "contact_trace_source": metadata.get("contact_trace_source"), "coverage_modes": coverage,
                "mount_status": "provisional; physical variant, interface and optical map not verified",
                "run_mode": mode, "review_scope": scope,
                "contact_poster_frame": peak_frame, "contact_poster_t": float(grid[peak_frame]),
                "fresh_frames": {"gel": int(gf.sum()), "body": int(bf.sum())},
                "output_sha256": sha(destination), "alignment_sha256": sha(args.out / "review_alignment.npz"),
                "codec": "CPU libx264 CRF20, yuv420p, faststart", "encode_command": command,
                "limitations": ["Body force and active compression use different filters/projection; their difference is not a measured rejected force.",
                                "Active force uses an assumed optical window and pressure distribution; no validated optical tactile or shear rendering.",
                                "Input hashes, frame timestamps/count/fps bind supplied files; source video identity is not inferred from pixels."]}
    (args.out / "review.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"video": str(destination), "frames": len(grid), "fresh_frames": manifest["fresh_frames"]}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--alignment", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-age-s", type=float, default=.15)
    a = p.parse_args()
    if not 0 < a.max_age_s <= 1:
        p.error("max-age-s must be between 0 and 1 second")
    run(a)
