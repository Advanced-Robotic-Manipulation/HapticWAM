"""Assemble a new immutable v1 copy with only gel-v2 and FINISH replacements."""

import ast
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

BASE = Path("/home/physicalai/phantom-icra-2027/sim/waffles")
OLD = BASE / "source_teacher_pick_place_v1"
OUT = BASE / "source_teacher_anchor_minimal_v5"
MANIFEST = BASE / "runs/teacher_success_anchor_v3/minimal_profile_source_manifest.json"
REPLACEMENTS = {
    "tools/sim/run_waffles.py": (
        "source_teacher_anchor_gel_v3",
        "b5184eb802ed2d99e1ac2f678fe485b810c72d878b0a52f330d4fec9423ca923",
    ),
    "tools/sim/gel_contact.py": (
        "source_teacher_anchor_gel_v3",
        "d7f4cb1d4606aa0ced6d360be6073498e5505460248d75cf6d04c7cf52853d6e",
    ),
    "phantom/sim/policy_adapter.py": (
        "source_teacher_anchor_finish_v3",
        "97adde55eb1b6e619054c45426e571ab08fe39f5f2da16e51b3b5a02022eae88",
    ),
    "phantom/sim/release_controller.py": (
        "source_teacher_anchor_finish_v3",
        "9719c8fc98d8eb390c67e81462fbf48f926d017b27bfe62f5eedeb51162c0284",
    ),
}
BASE_PINS = {
    "tools/sim/run_waffles.py": "355c6a76f00c55721ffad8744337863b6962b4777bb7df27ce5ff400dcba91a6",
    "tools/sim/gel_contact.py": "99791534245d6509a58c651d049a8fa80c981e19a53e7e462af581c23a66f300",
    "phantom/sim/policy_adapter.py": "3ff10500ccab0ebd747207f95bdf3049ae001fe3d98a3ba59489b809d208ef7d",
    "phantom/sim/release_controller.py": "16c8853989c1c9bc7a99cc949de8757e29f79ca7022fd6494f23ce6f2194d3b8",
}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot(root):
    return {
        str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*")) if p.is_file()
    }


def main():
    if OUT.exists() or MANIFEST.exists():
        raise FileExistsError(
            "Preserve existing source/manifest; destination must be new"
        )
    if OUT.resolve().is_relative_to(OLD.resolve()):
        raise ValueError("Destination may not be inside immutable base")
    before = snapshot(OLD)
    for rel, expected in BASE_PINS.items():
        assert before[rel] == expected, rel
    for rel, (donor, expected) in REPLACEMENTS.items():
        p = BASE / donor / rel
        assert sha(p) == expected, str(p)
        ast.parse(p.read_text())
    old_runner = (OLD / "tools/sim/run_waffles.py").read_text()
    gel_runner = (
        BASE / "source_teacher_anchor_gel_v3/tools/sim/run_waffles.py"
    ).read_text()
    assert gel_runner == old_runner.replace(
        'choices=["point", "manifold_patch"],',
        'choices=["point", "manifold_patch", "manifold_patch_v2"],',
        1,
    )
    # A real copy avoids writes through symlinks/hardlinks into frozen sources.
    shutil.copytree(OLD, OUT, symlinks=False, copy_function=shutil.copy2)
    for rel, (donor, _) in REPLACEMENTS.items():
        shutil.copy2(BASE / donor / rel, OUT / rel)
    after = snapshot(OUT)
    assert set(before) == set(after), "Unexpected source file addition/removal"
    changed = sorted(k for k in before if before[k] != after[k])
    assert changed == sorted(REPLACEMENTS), changed
    assert snapshot(OLD) == before, "Immutable base changed during assembly"
    assert not any(p.is_symlink() for p in OUT.rglob("*")), (
        "Output must be independent full copy"
    )
    for rel, (_, expected) in REPLACEMENTS.items():
        assert after[rel] == expected
    manifest = {
        "schema_version": 1,
        "variant": "immutable_v1_plus_gel_v2_and_finish_only",
        "assembled_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_source": str(OLD),
        "output_source": str(OUT),
        "copy_semantics": "Independent full copy, followed by exactly four hash-pinned replacements; no symlink or hardlink sharing",
        "file_counts": {
            "base": len(before),
            "output": len(after),
            "changed": len(changed),
            "unchanged": len(before) - len(changed),
        },
        "changed_files": {
            rel: {
                "base_sha256": before[rel],
                "source": str(BASE / REPLACEMENTS[rel][0] / rel),
                "output_sha256": after[rel],
            }
            for rel in changed
        },
        "base_unchanged_after_assembly": True,
        "source_sha256": after,
        "source_sha256_canonical_digest": hashlib.sha256(
            json.dumps(after, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "required_separate_profile": {
            "gel_contact_coverage": "manifold_patch_v2",
            "wrist_model": "contact_proxy",
            "terminal_veto_implementation": "fd4a032",
            "terminal_veto_feedback": "request_snapshot_historical",
            "placement_release_override": {"finish_after_release": True},
        },
        "pre_finish_semantics": "Identical v1 arm/gripper commands, native veto/latch, safety and rate limits until measured unloaded/open release dwell can finish. Diagnostics add completion fields. Gel-v2 changes tactile inputs and can therefore change policy behavior before release.",
        "finish_trigger": "Existing policy-commanded release with prior loaded latch; measured unloaded/open dwell plus accepted opening and measured TCP in unchanged clamp; no object scorer input",
        "finish_hold": "Measured achieved pose/open command held, replans disabled in adapter, pending proposal cleared; safety remains active. Frozen runner holds to full declared horizon unless actual stop.",
        "excluded_changes": [
            "gripper_contact_proxy",
            "live terminal veto",
            "current-delivery veto feedback",
            "native deploy shared-release hooks",
            "fixed wrist baseline",
            "latency schedule/override",
            "object state control",
            "physics or safety threshold changes",
        ],
        "runtime_or_hardware_launches": False,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
