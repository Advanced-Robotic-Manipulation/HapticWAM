"""F17 (2026-08-30 P0 #7): every number in start_poses.yaml comes from the
SAME episode set, and the loader refuses a file where it does not.

The 2026-08-28 file carried q_n = 17..37 against n = 250: the STOP hitbox, the
z no-go floor and the joint gate were fitted to 7% of the demos while
tcp_mean/tcp_std used all 250. These tests drive the real entry points —
`gen_start_poses.main` over a synthetic zarr dataset and `load_start_stats`
over both the generated and the shipped yaml.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
import zarr

from phantom.deploy import start_pose as sp
from tools import gen_start_poses as gsp

DOF = 6


def _write_stream(ep: Path, name: str, data: np.ndarray) -> None:
    g = zarr.open_group(str(ep / f"{name}.zarr"), mode="w")
    g.create_dataset("data", data=np.asarray(data, dtype=np.float64), chunks=(8,) + data.shape[1:])
    g.create_dataset("ts", data=np.arange(len(data), dtype=np.float64) / 10.0, chunks=(8,))


def _episode(root: Path, task: str, i: int, *, z_lo: float, x_lo: float,
             meta: dict | None = None) -> Path:
    """One synthetic episode; frame 0 is the start pose, frame 1 the extreme."""
    ep = root / task / f"ep_{task}_{i:04d}"
    ep.mkdir(parents=True)
    tcp = np.array([
        # start: every component varies, so no std is degenerate
        [-0.30 + 0.001 * i, -0.20 + 0.0007 * i, 0.30 + 0.0005 * i,
         -1.0 + 0.002 * i, -1.8 + 0.0015 * i, 1.5 + 0.001 * i],
        [x_lo, -0.35, z_lo, -1.0, -1.8, 1.5],                # envelope extreme
        [-0.25, 0.10, 0.40, -1.0, -1.8, 1.5],
    ])
    _write_stream(ep, "arm_tcp_pose", tcp)
    _write_stream(ep, "arm_q", np.tile(np.arange(DOF, dtype=np.float64), (3, 1)) + 0.001 * i)
    _write_stream(ep, "gripper", np.full((3, 1), 0.2 + 0.001 * i))
    m = {"episode": ep.name, "task": task, "success": True, "status": "finalized"}
    (ep / "meta.json").write_text(json.dumps({**m, **(meta or {})}))
    return ep


@pytest.fixture()
def dataset(tmp_path: Path) -> Path:
    """25 success episodes per task + traps the generator must exclude."""
    root = tmp_path / "tasks"
    for task in gsp.TASKS:
        for i in range(25):
            # one episode per task reaches much lower / further than the rest:
            # a subset that misses it produces a too-tight hitbox and floor
            _episode(root, task, i, z_lo=0.05 if i == 24 else 0.20,
                     x_lo=-0.55 if i == 24 else -0.40)
        # traps, all inside the SUCCESS task dir (this is how the recovery
        # intake symlinks land) — none may enter the stats
        _episode(root, task, 90, z_lo=-0.50, x_lo=-9.0,
                 meta={"success": False})
        _episode(root, task, 91, z_lo=-0.50, x_lo=-9.0,
                 meta={"failure_demo": True})
        _episode(root, task, 92, z_lo=-0.50, x_lo=-9.0,
                 meta={"tags": ["deliberate_failure", "undergrasp"]})
        _episode(root, task, 93, z_lo=-0.50, x_lo=-9.0,
                 meta={"task": task + "_fail"})
        # a whole *_fail task directory is never even globbed
        _episode(root, task + "_fail", 0, z_lo=-0.90, x_lo=-9.0,
                 meta={"task": task + "_fail"})
    return root


def test_generator_one_episode_set_behind_every_number(dataset: Path, tmp_path: Path):
    out = tmp_path / "start_poses.yaml"
    js = tmp_path / "stats.json"
    assert gsp.main([str(dataset), "--out", str(out), "--json-out", str(js)]) == 0
    raw = yaml.safe_load(out.read_text())
    assert set(raw["tasks"]) == set(gsp.TASKS)
    assert raw["n_episodes_total"] == 4 * 25
    for task, d in raw["tasks"].items():
        assert d["n"] == 25, task                 # the 5 traps are excluded
        assert d["q_n"] == d["n"], task           # the whole point of F17
        # the envelope/floor see the ONE far episode, not just the common ones
        assert d["tcp_z_min"] == pytest.approx(0.05, abs=1e-6), task
        assert d["tcp_min"][0] == pytest.approx(-0.55, abs=1e-6), task
        assert d["tcp_max"][1] == pytest.approx(0.10, abs=1e-6), task
        # and never the failure demos' absurd values
        assert d["tcp_min"][0] > -1.0 and d["tcp_z_min"] > 0.0, task
    per_task = json.loads(js.read_text())
    for task in gsp.TASKS:
        eps = per_task[task]["episodes"]
        assert len(eps) == len(set(eps)) == 25, task
        assert not any(e.endswith(("0090", "0091", "0092", "0093")) for e in eps), task


def test_generated_file_loads_and_gates(dataset: Path, tmp_path: Path):
    out = tmp_path / "start_poses.yaml"
    gsp.main([str(dataset), "--out", str(out)])
    stats = sp.load_start_stats(out)
    assert set(stats) == set(gsp.TASKS)
    st = stats["waffles"]
    assert st.n == 25 and st.q_mean.shape == (DOF,)
    # the joint gate is live on the generated stats
    sig, _ = sp.start_sigma_report(st, st.tcp_mean, st.gripper_mean, st.q_mean)
    assert np.all(sig < 1e-6)
    wrapped = st.q_mean.copy()
    wrapped[5] += 2 * np.pi
    sig, lines = sp.start_sigma_report(st, st.tcp_mean, st.gripper_mean, wrapped)
    assert sig.max() > 10 and "FULL-TURN" in lines


def test_generator_refuses_ragged_and_thin(dataset: Path, tmp_path: Path):
    # an episode whose arm_q is missing must not silently shorten Q alone
    bad = dataset / "egg" / "ep_egg_0000"
    for f in sorted((bad / "arm_q.zarr").rglob("*"), reverse=True):
        f.unlink() if f.is_file() else f.rmdir()
    (bad / "arm_q.zarr").rmdir()
    out = tmp_path / "start_poses.yaml"
    assert gsp.main([str(dataset), "--out", str(out)]) == 0
    d = yaml.safe_load(out.read_text())["tasks"]["egg"]
    assert d["n"] == d["q_n"] == 24        # the whole episode dropped, not just q

    thin = tmp_path / "thin"
    (thin / "Carton").mkdir(parents=True)
    with pytest.raises(AssertionError, match="refusing thin stats"):
        gsp.task_stats(str(thin), "Carton")


def test_loader_refuses_subset_joint_stats(dataset: Path, tmp_path: Path):
    out = tmp_path / "start_poses.yaml"
    gsp.main([str(dataset), "--out", str(out)])
    raw = yaml.safe_load(out.read_text())
    raw["tasks"]["egg"]["q_n"] = 7                  # the 2026-08-28 file's bug
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(sp.ThinStartStatsError, match="q_n=7 but n=25"):
        sp.load_start_stats(bad)
    # missing q_n on a file that HAS a joint block is the same failure
    del raw["tasks"]["egg"]["q_n"]
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(sp.ThinStartStatsError):
        sp.load_start_stats(bad)
    # explicit opt-out warns instead (offline analysis only)
    st = sp.load_start_stats(bad, allow_thin_q=True)
    assert st["egg"].n == 25


def test_loader_accepts_pre_joint_block_file(tmp_path: Path):
    """A file from before 2026-08-28 has no joint block at all — still loads."""
    raw = {"tasks": {"egg": {"n": 180,
                             "tcp_mean": [-0.35, -0.22, 0.26, -1.4, -1.75, 1.19],
                             "tcp_std": [0.02, 0.03, 0.03, 0.14, 0.11, 0.11],
                             "gripper_mean": 0.27, "gripper_std": 0.12}}}
    p = tmp_path / "old.yaml"
    p.write_text(yaml.safe_dump(raw))
    st = sp.load_start_stats(p)["egg"]
    assert st.q_mean is None and st.tcp_min is None


def test_shipped_start_poses_is_full_dataset():
    """The file run_deploy actually loads: full-dataset stats, q_n == n."""
    path = Path(sp.__file__).resolve().parents[2] / "configs" / "start_poses.yaml"
    raw = yaml.safe_load(path.read_text())
    stats = sp.load_start_stats()               # refuses if q_n != n
    assert set(stats) == {"Carton", "egg", "waffles", "whiteboard"}
    for task, d in raw["tasks"].items():
        assert d["q_n"] == d["n"] >= 250, task
        st = stats[task]
        assert st.tcp_min is not None and st.tcp_z_min is not None
        # the hitbox must contain the start distribution it is gating
        assert np.all(st.tcp_min <= st.tcp_mean[:3]) and np.all(st.tcp_mean[:3] <= st.tcp_max)
        assert st.tcp_z_min <= st.tcp_mean[2]
