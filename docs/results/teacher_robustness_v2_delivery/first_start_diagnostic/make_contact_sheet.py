#!/usr/bin/env python3
"""Lay out immutable extracted simulation frames, with explicit event labels."""

import base64
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent


def main():
    bundle = json.loads(Path(sys.argv[1]).read_text())
    frames = bundle["frames"]
    target = HERE / "frames"
    target.mkdir(exist_ok=True)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 17)
    small = ImageFont.truetype(font_path, 14)
    bold = ImageFont.truetype(font_path, 21)
    canvas = Image.new("RGB", (1024, 1808), "#101820")
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 12), "Corrected study: start 1787395928", font=bold, fill="white")
    draw.text(
        (16, 41),
        "Existing simulation frames; no real footage or hardware force calibration",
        font=small,
        fill="#ddd",
    )
    labels = [
        "Missed grasp; 56.2 mm at first additional close",
        "Housing / bin front: wrench stop",
        "Missed grasp; 64.1 mm at first additional close",
        "Housing / bin front: wrench stop",
        "Right finger / mat + bench: wrench stop",
        "Left finger / bin front: wrench stop",
        "Left finger / bin front: wrench stop",
        "Right finger / packet: wrench stop",
    ]
    for i, (record, label) in enumerate(zip(frames, labels)):
        data = base64.b64decode(record.pop("jpeg_base64"))
        assert hashlib.sha256(data).hexdigest() == record["jpeg_sha256"]
        path = target / (record["case_id"] + ".jpg")
        path.write_bytes(data)
        record["local_frame"] = str(path.relative_to(HERE))
        x, y = (i % 2) * 512, 76 + (i // 2) * 428
        with Image.open(path) as image:
            canvas.paste(image.resize((512, 384), Image.Resampling.LANCZOS), (x, y))
        policy, _, seed = record["case_id"].split("__")
        draw.rectangle((x, y, x + 511, y + 45), fill="#101820")
        draw.text((x + 8, y + 4), f"{policy} | {seed}", font=font, fill="white")
        draw.text(
            (x + 8, y + 25),
            f"Frame t={record['frame_time_s']:.3f}s; event t={record['event_time_s']:.3f}s",
            font=small,
            fill="#ddd",
        )
        draw.text((x + 8, y + 390), label, font=small, fill="#ffe0a0")
    path = HERE / "contact_sheet.jpg"
    canvas.save(path, quality=91)
    bundle["contact_sheet"] = {
        "path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    bundle["semantics"] = (
        "Last rendered scene sample at or before the first safety stop; first additional closing command instead for two horizon cases. Labels use independently logged contacts, not image inference. Actors may be occluded. No frame pixels were changed except resizing and text strips."
    )
    (HERE / "frames_manifest.json").write_text(json.dumps(bundle, indent=2) + "\n")


if __name__ == "__main__":
    main()
