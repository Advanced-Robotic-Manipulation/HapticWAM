#!/usr/bin/env python
"""Make a published pi0.5 checkpoint's processor pipeline loadable on lerobot 0.4.4.

`lerobot/pi05_base` was published against a lerobot that registered two
processor steps 0.4.4 does not have:

    policy_preprocessor.json   relative_actions_processor
    policy_postprocessor.json  absolute_actions_processor

`PolicyProcessorPipeline.from_pretrained` resolves every step through
`ProcessorStepRegistry` before looking at its config, so an unknown
`registry_name` is a hard ImportError even though BOTH steps are published
with `"enabled": false` -- i.e. they are no-ops for this checkpoint and
dropping them cannot change behaviour. pi0.5 here is trained on absolute
targets anyway: our `action` feature is already a base-frame pose DELTA
produced by the exporter, so lerobot must not apply a second relative
transform on top of it.

The PaliGemma tokenizer is the other blocker: the published preprocessor names
`google/paligemma-3b-pt-224`, which is a GATED repo. Point `--tokenizer` at a
local directory holding the tokenizer files (see docs/pi05_baseline.md for a
hash-verified public mirror) and the step is rewritten to load from disk.

`--single-camera` collapses `config.json`'s three published camera slots
(`base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`) to the ONE key our rig
actually has. This is a memory decision, and it is exactly behaviour-neutral:

  * a camera key the batch does not carry is appended by `modeling_pi05` as a
    -1 image with a ZERO `img_mask`, and `embed_prefix` turns those masks into
    the 2-D attention mask, so a blank camera's tokens are attended by nothing;
  * `position_ids = cumsum(pad_masks) - 1`, so a masked token does not advance
    the rotary position of anything after it either.

A blank slot is therefore pure cost: on a 4090 the three-slot layout OOMs at
batch 4 (20.7 GiB) while one slot fits batch 6 in ~17.5 GiB. Nothing in
`lerobot.policies.pi0*` matches the literal camera names, so keeping OUR key
(rather than renaming to `base_0_rgb` via `--rename_map`) also lets the deploy
adapter's checkpoint contract match `config.image_features` exactly.

Every edit keeps a `.orig` sibling and is idempotent -- re-running is a no-op.

Usage:
    python tools/patch_pi05_processors.py --policy-dir ~/lerobot/pi05_base \
        [--tokenizer ~/lerobot/paligemma_tokenizer] \
        [--single-camera observation.images.scene] [--check]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# steps published in lerobot/pi05_base that lerobot 0.4.4 does not register.
# Both are published disabled, so removing them is behaviour-preserving.
UNREGISTERED = {
    "policy_preprocessor.json": "relative_actions_processor",
    "policy_postprocessor.json": "absolute_actions_processor",
}


def patch_file(path: Path, drop: str, tokenizer: Path | None, check: bool) -> list[str]:
    if not path.exists():
        return [f"{path.name}: MISSING"]
    cfg = json.loads(path.read_text())
    notes: list[str] = []
    changed = False

    steps = cfg.get("steps", [])
    kept = [s for s in steps if s.get("registry_name") != drop]
    if len(kept) != len(steps):
        dropped = len(steps) - len(kept)
        for s in steps:
            if s.get("registry_name") == drop and s.get("config", {}).get("enabled", False):
                raise SystemExit(
                    f"{path.name}: refusing to drop {drop!r} -- it is ENABLED in this "
                    f"checkpoint, so removing it would change the action convention"
                )
        cfg["steps"] = kept
        changed = True
        notes.append(f"{path.name}: dropped {dropped}x {drop} (unregistered, enabled=false)")
    else:
        notes.append(f"{path.name}: no {drop} step (already patched or not published)")

    if tokenizer is not None:
        for s in cfg.get("steps", []):
            if s.get("registry_name") == "tokenizer_processor":
                cur = s["config"].get("tokenizer_name")
                if cur != str(tokenizer):
                    s["config"]["tokenizer_name"] = str(tokenizer)
                    changed = True
                    notes.append(f"{path.name}: tokenizer_name {cur} -> {tokenizer}")
                else:
                    notes.append(f"{path.name}: tokenizer_name already {tokenizer}")

    if changed and not check:
        orig = path.with_suffix(path.suffix + ".orig")
        if not orig.exists():
            orig.write_text(path.read_text())
            notes.append(f"{path.name}: backed up to {orig.name}")
        path.write_text(json.dumps(cfg, indent=2))
    elif changed and check:
        notes.append(f"{path.name}: WOULD CHANGE (--check, nothing written)")
    return notes


def patch_camera(path: Path, key: str, check: bool) -> list[str]:
    """Collapse config.json's VISUAL input features to the single key `key`."""
    cfg = json.loads(path.read_text())
    feats = cfg.get("input_features", {})
    visual = [k for k, v in feats.items() if v.get("type") == "VISUAL"]
    if not visual:
        raise SystemExit(f"{path.name}: no VISUAL input features to collapse")
    if visual == [key]:
        return [f"{path.name}: already a single camera {key}"]

    shape = feats[visual[0]]["shape"]
    for k in visual:
        if feats[k]["shape"] != shape:
            raise SystemExit(f"{path.name}: camera {k} shape {feats[k]['shape']} != {shape}")
        del feats[k]
    feats[key] = {"type": "VISUAL", "shape": shape}
    notes = [f"{path.name}: cameras {visual} -> [{key}] shape {shape}"]

    if check:
        notes.append(f"{path.name}: WOULD CHANGE (--check, nothing written)")
        return notes
    orig = path.with_suffix(path.suffix + ".orig")
    if not orig.exists():
        orig.write_text(json.dumps(json.loads(path.read_text()), indent=4))
        notes.append(f"{path.name}: backed up to {orig.name}")
    path.write_text(json.dumps(cfg, indent=4))
    return notes


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policy-dir", type=Path, required=True,
                   help="a downloaded pi05 checkpoint directory")
    p.add_argument("--tokenizer", type=Path, default=None,
                   help="local dir with the PaliGemma tokenizer files")
    p.add_argument("--single-camera", default=None, metavar="KEY",
                   help="collapse config.json's camera slots to this one key, "
                        "e.g. observation.images.scene")
    p.add_argument("--check", action="store_true", help="report, do not write")
    args = p.parse_args()

    policy_dir = args.policy_dir.expanduser()
    tokenizer = args.tokenizer.expanduser() if args.tokenizer else None
    if tokenizer is not None and not (tokenizer / "tokenizer.json").exists():
        raise SystemExit(f"no tokenizer.json under {tokenizer}")

    for name, drop in UNREGISTERED.items():
        for note in patch_file(policy_dir / name, drop, tokenizer, args.check):
            print(note)
    if args.single_camera:
        for note in patch_camera(policy_dir / "config.json", args.single_camera, args.check):
            print(note)
    print("OK")


if __name__ == "__main__":
    main()
