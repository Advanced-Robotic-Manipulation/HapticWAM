"""tools/hub/fetch_episode.py — episode id -> archive resolution.

Everything here runs against LOCAL fixture copies of the hub index files
(tests/fixtures/hub/*.jsonl); nothing touches the network. The entry point is
exercised through `main(..., --dry-run)` with the two hub accessors
(`_index`, `_dirs`) redirected at the fixtures, so the CLI wiring is covered
too, not just the pure resolvers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools" / "hub"))

import fetch_episode as fe  # noqa: E402

FIX = REPO / "tests" / "fixtures" / "hub"
TELEOP = fe.read_jsonl(FIX / "teleop_index.jsonl")
SIM = fe.read_jsonl(FIX / "sim_index.jsonl")
ROLL_DAYS = fe.read_jsonl(FIX / "rollouts_rig_deploy_index.jsonl")
ROLL_FILES = fe.read_jsonl(FIX / "rollouts_index.jsonl")


# --------------------------------------------------------------------------
# episode id parsing
# --------------------------------------------------------------------------

@pytest.mark.parametrize("ep,task", [
    ("ep_waffles_1785592739_002", "waffles"),
    ("ep_waffles_fail_1785592739_002", "waffles_fail"),          # task has an underscore
    ("ep_student_Carton_1789493413_000", "student_Carton"),      # deploy: policy prefix
    ("ep_whiteboard_1785600000_017", "whiteboard"),
])
def test_parse_episode_id(ep, task):
    assert fe.parse_episode_id(ep)[0] == task


def test_parse_episode_id_rejects_junk():
    for bad in ("waffles_1785592739_002", "ep_waffles", "ep_waffles_12_002"):
        with pytest.raises(ValueError):
            fe.parse_episode_id(bad)


# --------------------------------------------------------------------------
# teleop: episode -> task shard
# --------------------------------------------------------------------------

def test_resolve_teleop_maps_episode_to_its_task_shard():
    r = fe.resolve_teleop("ep_waffles_1785592739_002", TELEOP)
    assert (r.repo, r.kind, r.path) == (fe.TELEOP_REPO, "tar", "waffles.tar.zst")
    assert r.member_prefix == "waffles/ep_waffles_1785592739_002/"
    assert r.compression == "zst"


def test_resolve_teleop_failure_demo_shard():
    r = fe.resolve_teleop("ep_waffles_fail_1785592739_002", TELEOP)
    assert r.path == "waffles_fail.tar.zst"
    assert r.member_prefix == "waffles_fail/ep_waffles_fail_1785592739_002/"


def test_resolve_teleop_first_takes_a_whole_shard_and_stops_early():
    r = fe.resolve_teleop("first", TELEOP)
    assert r.path in fe.teleop_shards(TELEOP)
    assert r.episode == "" and r.member_prefix == ""


def test_resolve_teleop_unknown_task_names_the_batch_archive():
    with pytest.raises(LookupError) as e:
        fe.resolve_teleop("ep_bananas_1785592739_002", TELEOP)
    assert "batch_20260822.tar.zst" in str(e.value)
    assert "waffles.tar.zst" in str(e.value)          # lists what IS there


def test_resolve_teleop_archive_override():
    r = fe.resolve_teleop("ep_waffles_1785592739_002", TELEOP,
                          archive="batch_20260822.tar.zst")
    assert r.path == "batch_20260822.tar.zst"


def test_teleop_shards_exclude_the_batch_and_nested_files():
    assert "batch_20260822.tar.zst" not in fe.teleop_shards(TELEOP)
    assert "waffles.tar.zst" in fe.teleop_shards(TELEOP)


def test_resolve_teleop_loose_points_at_the_raw_repo():
    r = fe.resolve_teleop_loose("ep_egg_1785592739_002")
    assert (r.repo, r.kind) == (fe.TELEOP_LOOSE_REPO, "loose")
    assert r.path == "tasks/egg/ep_egg_1785592739_002"


# --------------------------------------------------------------------------
# sim: unit -> per-episode tar
# --------------------------------------------------------------------------

def test_resolve_sim_by_unit_basename():
    r = fe.resolve_sim("ep_sim_egg_egg2_ep0001__ep_student_egg_1788970209_002", SIM)
    assert r.path == ("sim_expert_20260914/tasks/egg/"
                      "ep_sim_egg_egg2_ep0001__ep_student_egg_1788970209_002.tar")
    assert r.compression == ""          # plain tar


def test_resolve_sim_by_full_unit_path():
    unit = "tasks/waffles/ep_sim_waffles_v5__ep_waffles_1785593867_004_open"
    assert fe.resolve_sim(unit, SIM).path.endswith(unit + ".tar")


def test_resolve_sim_first_is_the_smallest_successful_task_unit():
    r = fe.resolve_sim("first", SIM)
    # the 98 MB egg unit is smaller but success=False; the raw_trials campaign
    # and the aside_duplicates copy are not in tasks/
    assert r.path.endswith("ep_sim_egg_egg2_ep0001__ep_student_egg_1788970209_002.tar")


def test_resolve_sim_unknown_unit():
    with pytest.raises(LookupError):
        fe.resolve_sim("ep_sim_nope", SIM)


# --------------------------------------------------------------------------
# rollouts: day -> tar
# --------------------------------------------------------------------------

def test_resolve_rollouts_prefers_the_uncompressed_repack_of_a_day():
    r = fe.resolve_rollouts("deploy_20260813", ROLL_DAYS, ROLL_FILES)
    assert r.path == "zarr_rollouts/deploy_20260813.tar"   # not the .tar.zst twin
    assert r.compression == ""


def test_resolve_rollouts_day_from_the_rig_deploy_index():
    r = fe.resolve_rollouts("deploy_20260911", ROLL_DAYS, ROLL_FILES)
    assert r.path == "rig_deploy/deploy_20260911.tar"
    assert "148" in r.note                                  # takes in the pack


def test_resolve_rollouts_take_needs_its_day():
    with pytest.raises(LookupError) as e:
        fe.resolve_rollouts("ep_student_waffles_1789000000_003", ROLL_DAYS, ROLL_FILES)
    assert "--day" in str(e.value)
    r = fe.resolve_rollouts("ep_student_waffles_1789000000_003", ROLL_DAYS, ROLL_FILES,
                            day="deploy_20260909")
    assert r.path == "zarr_rollouts/deploy_20260909.tar"
    assert r.member_prefix == "deploy_20260909/ep_student_waffles_1789000000_003/"


def test_resolve_rollouts_first_is_the_smallest_pack():
    r = fe.resolve_rollouts("first", ROLL_DAYS, ROLL_FILES)
    assert r.path in ("zarr_rollouts/deploy_20260813.tar", "deploy_20260813.tar.zst")


def test_resolve_rollouts_unknown_day_lists_the_known_ones():
    with pytest.raises(LookupError) as e:
        fe.resolve_rollouts("deploy_19990101", ROLL_DAYS, ROLL_FILES)
    assert "deploy_20260813" in str(e.value)


# --------------------------------------------------------------------------
# rig + samples: loose paths
# --------------------------------------------------------------------------

RIG_TREE = ["20260915_experiment/ep_student_Carton_1789493413_000",
            "20260915_experiment/ep_teacher_egg_1789495000_004",
            "20260915_experiment_extra/ep_student_waffles_1789496000_002"]


def test_resolve_rig_loose():
    r = fe.resolve_rig("ep_teacher_egg_1789495000_004", RIG_TREE)
    assert (r.kind, r.repo) == ("loose", fe.RIG_REPO)
    assert r.path == "20260915_experiment/ep_teacher_egg_1789495000_004"


def test_resolve_rig_unknown():
    with pytest.raises(LookupError):
        fe.resolve_rig("ep_nope_1789495000_004", RIG_TREE)


def test_resolve_samples():
    dirs = ["samples/waffles/ep_waffles_1785592739_002",
            "samples/egg/ep_egg_1785600000_003"]
    r = fe.resolve_samples("teleop", "ep_egg_1785600000_003", dirs)
    assert (r.kind, r.path) == ("loose", "samples/egg/ep_egg_1785600000_003")
    assert fe.resolve_samples("teleop", "first", dirs).path.startswith("samples/")


# --------------------------------------------------------------------------
# entry point (dry run, hub accessors redirected at the fixtures)
# --------------------------------------------------------------------------

@pytest.fixture
def offline_hub(monkeypatch):
    def _index(repo, path):
        if repo == fe.TELEOP_REPO:
            return TELEOP
        if repo == fe.ROLLOUTS_REPO:
            return ROLL_DAYS if path.startswith("rig_deploy/") else ROLL_FILES
        if repo == fe.SIM_REPO:
            return SIM
        raise AssertionError(f"unexpected index request {repo}/{path}")

    def _dirs(repo, prefix):
        if repo == fe.SIM_REPO and prefix == "":
            return ["sim_expert_20260912", "sim_expert_20260914"]
        if repo == fe.RIG_REPO:
            return [d for d in RIG_TREE if d.startswith(prefix + "/")]
        return []

    monkeypatch.setattr(fe, "_index", _index)
    monkeypatch.setattr(fe, "_dirs", _dirs)


@pytest.mark.parametrize("argv,expect", [
    (["--dataset", "teleop", "--episode", "ep_waffles_1785592739_002"],
     "waffles.tar.zst"),
    (["--dataset", "sim", "--episode", "first"],
     "sim_expert_20260914/tasks/egg/"),
    (["--dataset", "rollouts", "--episode", "deploy_20260813"],
     "zarr_rollouts/deploy_20260813.tar"),
    (["--dataset", "rig", "--episode", "ep_student_Carton_1789493413_000"],
     "20260915_experiment/ep_student_Carton_1789493413_000"),
])
def test_main_dry_run_resolves_without_network(offline_hub, capsys, argv, expect):
    assert fe.main([*argv, "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert expect in out
    assert "resolved" in out


def test_main_dry_run_reports_an_unresolvable_episode(offline_hub):
    with pytest.raises(LookupError):
        fe.main(["--dataset", "teleop", "--episode", "ep_bananas_1785592739_002",
                 "--dry-run"])
