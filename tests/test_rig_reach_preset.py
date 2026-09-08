"""Exercise the operator menu using fake launch/probe programs, no devices."""
import os
from pathlib import Path
import subprocess

import pytest


REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("system,preset,fixed", [
    ("teacher", "", True), ("teacher", "1", False),
    ("student", "", False), ("teacher", "5", True),
])
def test_menu_selects_reach_fix_and_uses_same_checkout(tmp_path, system, preset, fixed):
    base = tmp_path / "rig"
    repo = base / "phantom"
    ckpt = repo / "runs/model.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"fixture")
    (base / "MODELS.tsv").write_text(f"model\truns/model.pt\tfixture\t{system}\n")
    probe = repo / ".venv/bin/python"
    probe.parent.mkdir(parents=True)
    probe.write_text("#!/bin/bash\nexit 1\n")
    probe.chmod(0o755)
    launch = repo / "tools/rig/GO_ANY.sh"
    launch.parent.mkdir(parents=True)
    launch.write_text('#!/bin/bash\nprintf "TRACKED|%s|%s|%s|%s\\n" "$SYSTEM" "$CKPT" "$EXTRA" "$*"\n')
    (base / "GO_ANY.sh").write_text('#!/bin/bash\necho STALE_COPY\nexit 9\n')
    result = subprocess.run(
        ["bash", str(REPO / "tools/rig/PICK.sh")],
        input=f"\n{preset}\n\n\n1\n\n\n", capture_output=True, text=True,
        env={**os.environ, "PHANTOM_RIG_BASE": str(base)}, timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    command = next(line for line in result.stdout.splitlines() if line.startswith("TRACKED|"))
    assert "STALE_COPY" not in result.stdout
    assert f"TRACKED|{system}|runs/model.pt|" in command
    assert ("--servo-reach-profile bounded_v1" in command) is fixed
    assert "--max-play-steps 12" in command and "--seed 101" in command
    assert command.endswith("|waffles 1 1 1.0")
