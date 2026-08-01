"""collect/offload.py — verified move to the external drive."""

from pathlib import Path

from phantom.data_collect.offload import offload_session


def _fake_session(root: Path, n_eps=2, n_files=3) -> Path:
    staging = root / "20260722_120000_task"
    for e in range(n_eps):
        ep = staging / f"ep_task_{e:03d}"
        (ep / "sub").mkdir(parents=True)
        for f in range(n_files):
            (ep / f"stream_{f}.bin").write_bytes(bytes([e, f]) * 100)
        (ep / "sub" / "meta.json").write_text("{}")
    return staging


def test_offload_moves_and_deletes_local(tmp_path):
    staging = _fake_session(tmp_path / "staging")
    drive = tmp_path / "drive"
    drive.mkdir()
    events = []
    res = offload_session(staging, drive, verify="size",
                          progress=lambda d, t, n: events.append((d, t, n)))
    assert res.ok and res.episodes == 2 and res.bytes_moved > 0
    dest = drive / staging.name
    assert sorted(p.name for p in dest.glob("ep_*")) == \
        ["ep_task_000", "ep_task_001"]
    assert (dest / "ep_task_000" / "sub" / "meta.json").exists()
    # local copies are gone after the verified move
    assert list(staging.glob("ep_*")) == []
    assert events[-1][0] == events[-1][1] == 2


def test_missing_drive_keeps_local(tmp_path):
    staging = _fake_session(tmp_path / "staging")
    res = offload_session(staging, tmp_path / "no_such_drive")
    assert not res.ok and "not found" in res.error
    assert len(list(staging.glob("ep_*"))) == 2      # untouched


def test_sha256_verification(tmp_path):
    staging = _fake_session(tmp_path / "staging")
    drive = tmp_path / "drive"
    drive.mkdir()
    res = offload_session(staging, drive, verify="sha256")
    assert res.ok and res.episodes == 2


def test_keep_local(tmp_path):
    staging = _fake_session(tmp_path / "staging")
    drive = tmp_path / "drive"
    drive.mkdir()
    res = offload_session(staging, drive, keep_local=True)
    assert res.ok
    assert len(list(staging.glob("ep_*"))) == 2      # kept


def test_partial_previous_offload_is_redone(tmp_path):
    staging = _fake_session(tmp_path / "staging")
    drive = tmp_path / "drive"
    stale = drive / staging.name / "ep_task_000"
    stale.mkdir(parents=True)
    (stale / "half.bin").write_bytes(b"stale")       # partial earlier attempt
    res = offload_session(staging, drive)
    assert res.ok
    assert not (drive / staging.name / "ep_task_000" / "half.bin").exists()
    assert (drive / staging.name / "ep_task_000" / "stream_0.bin").exists()


def test_empty_staging_is_ok(tmp_path):
    staging = tmp_path / "empty"
    staging.mkdir()
    assert offload_session(staging, tmp_path / "drive").ok


def test_same_device_guard(tmp_path):
    """Review finding: an unmounted Linux mountpoint is a plain local dir —
    the 'offload' would land on the local disk and delete the local copy."""
    staging = _fake_session(tmp_path / "staging")
    drive = tmp_path / "drive"                    # same filesystem as staging
    drive.mkdir()
    res = offload_session(staging, drive, require_separate_device=True)
    assert not res.ok and "SAME filesystem" in res.error
    assert len(list(staging.glob("ep_*"))) == 2   # untouched


def test_insufficient_space_reported(tmp_path):
    staging = _fake_session(tmp_path / "staging")
    drive = tmp_path / "drive"
    drive.mkdir()
    res = offload_session(staging, drive, min_free_gb=10 ** 9)   # absurd reserve
    assert not res.ok and "free" in res.error
    assert len(list(staging.glob("ep_*"))) == 2
