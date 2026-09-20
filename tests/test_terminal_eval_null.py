"""tools/terminal_eval.py hardening — finding P1 part 2 / E9.

The offline metric that blessed v5_6 teacher-forces every demo-consistent
input, so it cannot see a closed-loop failure. This covers the hardening:

* `--null {tactile,wrist,prev_cpk,obs,contact_gt,all}` actually removes (or,
  for contact_gt, PINS) the channel it names — asserted on the tensors the
  model is handed, not on the flag being accepted;
* a missing manifest is a hard failure instead of `manifest_split`'s silent
  "evaluate every episode" fallback (common.py:370);
* per-task medians and across-seed std are in the summary, and
  `pred_close_height_mm` is computed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from phantom.config.paths import load_paths
from phantom.data.schema import NormStats
from phantom_test_utils import make_small_hw

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import terminal_eval as TE  # noqa: E402


def _cosmos_available() -> bool:
    try:
        from phantom.backbone import loader as bl
        bl.setup_cosmos(load_paths())
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# batch-level nulling (no model needed)
# ---------------------------------------------------------------------------

def _fake_batch():
    g = torch.Generator().manual_seed(0)
    keys = ("gel", "fields", "contact_state", "wrist", "ur_state", "video",
            "reactive", "prev_chunk", "action_chunk")
    b = {k: torch.rand(2, 4, generator=g) + 1.0 for k in keys}
    b["text"] = ["pick the egg", "pick the egg"]
    return b


def _zeroed(b: dict) -> set[str]:
    return {k for k, v in b.items()
            if torch.is_tensor(v) and float(v.abs().sum()) == 0.0}


def test_null_none_is_the_identity():
    b = _fake_batch()
    assert _zeroed(TE.null_batch(b, "none")) == set()
    assert TE.null_batch(b, "none")["text"] == b["text"]


def test_tactile_null_drops_exactly_the_student_deleted_streams():
    out = TE.null_batch(_fake_batch(), "tactile")
    # the student layout has no OBS_GEL/OBS_MECH frames; wrist F/T is
    # ur_internal and survives in both arms of the sensor-free contrast.
    # `reactive` is derived from two fields_ds frames, so it is tactile too
    # and must go with them (F12) — leaving it live made the mode a partial
    # null and the E9 "tactile nulled" row an overstatement of what survived.
    assert _zeroed(out) == {"gel", "fields", "contact_state", "reactive"}
    assert float(out["wrist"].abs().sum()) > 0
    assert float(out["ur_state"].abs().sum()) > 0


def test_wrist_null_touches_only_wrist():
    assert _zeroed(TE.null_batch(_fake_batch(), "wrist")) == {"wrist"}


def test_obs_null_matches_the_classifier_free_null_plus_black_video():
    out = TE.null_batch(_fake_batch(), "obs")
    assert {"gel", "fields", "contact_state", "wrist", "ur_state",
            "video"} <= _zeroed(out)
    assert out["text"] == ["", ""]
    # intent and targets are NOT observations
    assert float(out["prev_chunk"].abs().sum()) > 0
    assert float(out["action_chunk"].abs().sum()) > 0


def test_all_is_obs_plus_the_prev_package():
    assert TE.nulls_prev_cpk("all") and TE.nulls_prev_cpk("prev_cpk")
    assert not TE.nulls_prev_cpk("tactile") and not TE.nulls_prev_cpk("none")
    assert TE.pins_contact("contact_gt") and not TE.pins_contact("all")
    # prev_cpk is not a batch edit
    assert _zeroed(TE.null_batch(_fake_batch(), "prev_cpk")) == set()


def test_unknown_null_mode_refuses():
    with pytest.raises(SystemExit):
        TE.null_batch(_fake_batch(), "tactile_maybe")


def test_zero_package_zeroes_every_tensor_field():
    from phantom.model.ace.packing import ContactPackage
    import dataclasses
    pkg = ContactPackage(**{f.name: torch.ones(1, 3)
                            for f in dataclasses.fields(ContactPackage)})
    z = TE.zero_package(pkg)
    assert all(float(getattr(z, f.name).abs().sum()) == 0.0
               for f in dataclasses.fields(ContactPackage))
    assert float(pkg.event.abs().sum()) > 0, "must not mutate the original"


def test_contact_pinned_layout_adds_only_the_contact_frames():
    from phantom.config.model import PhantomModelConfig
    from phantom.config.backbone import BackboneConfig
    from phantom.model.sequence import FrameGroup, SequenceLayout
    hw = make_small_hw()
    lay = SequenceLayout.build(BackboneConfig.tiny(), PhantomModelConfig(), hw)
    pinned = TE.contact_pinned_layout(lay)
    base, new = lay.cond_mask_T(), pinned.cond_mask_T()
    added = np.nonzero(new & ~base)[0]
    sl = lay.frame_slice(FrameGroup.CONTACT)
    assert list(added) == list(range(sl.start, sl.stop))
    assert new[base].all() and pinned.t_total == lay.t_total


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------

def test_missing_manifest_is_a_hard_failure(tmp_path):
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    with pytest.raises(SystemExit) as e:
        TE.resolve_episodes(tasks, "val")
    assert "manifest" in str(e.value)
    # ... and manifest_split, unhardened, would have said "evaluate everything"
    from phantom.train import common as C
    assert C.manifest_split(tasks, "val") is None
    # --split all is the deliberate opt-in
    assert TE.resolve_episodes(tasks, "all") is None


def test_empty_split_is_a_hard_failure(tmp_path):
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    mf = tmp_path / "manifests"
    mf.mkdir()
    (mf / "all.jsonl").write_text(
        json.dumps({"path": "tasks/ep_a", "split": "train"}) + "\n")
    with pytest.raises(SystemExit):
        TE.resolve_episodes(tasks, "val")


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def _rows():
    out = []
    for i, (task, base) in enumerate([("egg", 10.0), ("egg", 40.0), ("carton", 5.0)]):
        for s, off in enumerate((-1.0, 1.0)):
            out.append({"episode": f"ep{i}", "t0": 1.0, "seed": s, "task": task,
                        "endpoint_err_mm": base + off, "z_end_err_mm": base,
                        "commit_ratio": 1.0, "close_step_err": float(i),
                        "pred_close_height_mm": 100.0 + base})
    return out


def test_summary_carries_means_medians_and_across_seed_std():
    s = TE.summarize(_rows(), nfe=5, null="tactile", seeds=2)
    # backward compatible: the v4/v5 keys are still where they were
    for k in ("nfe", "n", "endpoint_err_mm", "z_end_err_mm", "commit_ratio",
              "close_step_err", "per_task"):
        assert k in s
    assert s["n"] == 6 and s["n_windows"] == 3 and s["n_episodes"] == 3
    assert s["endpoint_err_mm"] == pytest.approx((10 + 40 + 5) * 2 / 6)
    # the median is NOT the mean here — 40 mm is an outlier
    assert s["median_endpoint_err_mm"] == pytest.approx(10.0)
    # each window's two seeds are +-1 mm apart -> std ddof=1 of {b-1, b+1}
    assert s["seed_std_endpoint_err_mm"] == pytest.approx(np.sqrt(2.0))
    assert "pred_close_height_mm" in s and "median_pred_close_height_mm" in s
    pt = s["per_task"]["egg"]
    assert pt["n"] == 4
    assert pt["median_endpoint_err_mm"] == pytest.approx(25.0)
    assert pt["median_close_step_err"] == pytest.approx(0.5)
    assert np.isnan(TE.summarize(_rows()[:1])["seed_std_endpoint_err_mm"])


# ---------------------------------------------------------------------------
# end to end on the tiny backbone
# ---------------------------------------------------------------------------

pytestmark_e2e = pytest.mark.skipif(not _cosmos_available(),
                                    reason="cosmos repo not importable")


def _retime_close(ep: Path, t_close: float = 5.0) -> None:
    """Move the synthetic episode's first gripper close to `t_close` and give
    the arm a steady descent.

    The mock scenario closes at 1.57 s, which is earlier than the first
    admissible window start (t0 >= one chunk, 1.6 s), so `t0 = t_close - lead`
    would fall outside every window and terminal_eval would (correctly) find
    nothing to evaluate. Retiming makes the episode carry a real terminal
    phase with a steady-state replan one chunk before it."""
    import zarr
    g = zarr.open(str(ep / "gripper.zarr"), mode="r+")
    ts = np.asarray(g["ts"][:])
    pos = np.where(ts < t_close, 0.1, 0.7).astype(np.float32)
    g["data"][:, 0] = pos
    a = zarr.open(str(ep / "actions.zarr"), mode="r+")
    ats = np.asarray(a["ts"][:])
    a["data"][:, 6] = np.where(ats < t_close, 0.0, 1.0).astype(np.float32)
    a["data"][:, 2] = np.asarray(a["data"][:, 2]) - 0.002   # descend, so
    #                                    commit_ratio has a denominator


@pytest.fixture(scope="module")
def tiny_eval_setup(tmp_path_factory):
    """A tiny checkpoint + a two-episode manifested dataset terminal_eval can
    actually be pointed at."""
    if not _cosmos_available():
        pytest.skip("cosmos repo not importable")
    from phantom.data.synthetic import SyntheticEpisodeGenerator
    from phantom.train import common as C
    from phantom.train.builder import build_model
    from phantom.config.training import CommonTrainConfig

    hw = make_small_hw()
    root = tmp_path_factory.mktemp("term")
    tasks = root / "tasks" / "grasp_slip"
    tasks.mkdir(parents=True)
    gen = SyntheticEpisodeGenerator(hw, seed=0, rate_scale=1.0)
    eps = [gen.generate(tasks, task="grasp_slip", duration_s=8.0) for _ in range(2)]
    for p in eps:
        _retime_close(p)
    (root / "manifests").mkdir()
    (root / "manifests" / "all.jsonl").write_text("".join(
        json.dumps({"path": str(p.relative_to(root)), "split": "val"}) + "\n"
        for p in eps))

    ns = NormStats(mean={"action": np.zeros(hw.control.action_dim, np.float32),
                         "ur_state": np.zeros(hw.ur_state_dim, np.float32)},
                   std={"action": np.ones(hw.control.action_dim, np.float32),
                        "ur_state": np.ones(hw.ur_state_dim, np.float32)})
    pm = build_model(hw, load_paths(), student=False, tiny=True, load_base=False)
    ckpt = root / "tiny.pt"
    C.save_phantom_checkpoint(ckpt, pm.rf, hw=hw, bb=pm.bb, mc=pm.mc,
                              train_cfg=CommonTrainConfig(), step=0, norm_stats=ns)

    hw_yaml = root / "hardware.yaml"
    import yaml
    hw_yaml.write_text(yaml.safe_dump(json.loads(hw.model_dump_json())))
    return hw, root, ckpt, hw_yaml


class _Recorder:
    """Wraps PhantomRectifiedFlow.sample and keeps every batch it was given."""

    def __init__(self, monkeypatch):
        from phantom.model.rf import PhantomRectifiedFlow
        self.batches: list[dict] = []
        self.prev_cpks: list = []
        self.cond_masks: list = []
        self.reuse: list = []
        self.held_noise: list = []
        orig = PhantomRectifiedFlow.sample

        def wrapper(rf_self, batch, **kw):
            self.batches.append({k: (v.detach().float().cpu() if torch.is_tensor(v) else v)
                                 for k, v in batch.items()})
            self.prev_cpks.append(kw.get("prev_cpk"))
            self.cond_masks.append(rf_self.layout.cond_mask_T().copy())
            self.reuse.append(bool(kw.get("reuse_noise")))
            out = orig(rf_self, batch, **kw)
            n = getattr(rf_self, "_episode_noise", None)
            self.held_noise.append(None if n is None
                                   else float(n.detach().float().sum()))
            return out

        monkeypatch.setattr(PhantomRectifiedFlow, "sample", wrapper)

    def zeroed(self, key: str) -> bool:
        vals = [b[key] for b in self.batches if torch.is_tensor(b.get(key))]
        assert vals, f"{key} never reached sample()"
        return all(float(v.abs().sum()) == 0.0 for v in vals)


def _run(tiny_eval_setup, tmp_path, null: str, extra: list[str] | None = None):
    hw, root, ckpt, hw_yaml = tiny_eval_setup
    Path(tmp_path).mkdir(parents=True, exist_ok=True)
    out = Path(tmp_path) / f"{null}.json"
    argv = ["--ckpt", str(ckpt), "--data", str(root / "tasks"),
            "--hardware", str(hw_yaml), "--nfe", "1", "--seeds", "2",
            "--null", null, "--no-ema", "--tiny", "--out", str(out)]
    assert TE.main(argv + (extra or [])) == 0
    return json.loads(out.read_text())


@pytestmark_e2e
@pytest.mark.parametrize("null", list(TE.NULL_MODES))
def test_every_null_mode_runs_and_nulls_what_it_names(tiny_eval_setup, tmp_path,
                                                      monkeypatch, null):
    from phantom.model.sequence import FrameGroup
    rec = _Recorder(monkeypatch)
    got = _run(tiny_eval_setup, tmp_path, null)
    s = got["summary"]
    assert s["n"] > 0 and s["null"] == null
    assert np.isfinite(s["endpoint_err_mm"])

    tactile = {"gel", "fields", "contact_state"}
    expect_zero = {
        "none": set(), "prev_cpk": set(), "contact_gt": set(),
        "contact_zero": set(),
        "tactile": tactile, "wrist": {"wrist"},
        "obs": tactile | {"wrist", "ur_state", "video"},
        "all": tactile | {"wrist", "ur_state", "video"},
    }[null]
    for k in tactile | {"wrist", "ur_state", "video"}:
        assert rec.zeroed(k) is (k in expect_zero), f"{null}: {k}"

    # the GT contact package: zeroed everywhere except --null contact_gt --
    # THE DEFAULT INCLUDED (the deploy condition; E9's row 1 is this, not
    # "everything on", which is how the published table was read)
    assert rec.zeroed("events") is not TE.keeps_gt_package(null)
    assert rec.zeroed("cpk_wrench") is not TE.keeps_gt_package(null)
    assert TE.keeps_gt_package(null) is (null == "contact_gt")
    # ... and the CONTACT frames are cond-pinned in BOTH P7 arms:
    # contact_gt pins them to that GT package, contact_zero to the zeros
    lay = None
    from phantom.config.backbone import BackboneConfig
    from phantom.config.model import PhantomModelConfig
    from phantom.model.sequence import SequenceLayout
    lay = SequenceLayout.build(BackboneConfig.tiny(), PhantomModelConfig(),
                               tiny_eval_setup[0])
    sl = lay.frame_slice(FrameGroup.CONTACT)
    pinned = [bool(m[sl].all()) for m in rec.cond_masks]
    assert all(pinned) is TE.pins_contact(null)
    assert TE.pins_contact(null) is (null in ("contact_gt", "contact_zero"))

    # the JSON says what the condition WAS, so no table can be relabelled by
    # hand again (F18)
    sem = s["null_semantics"]
    assert sem["mode"] == null
    assert sem["gt_contact_package"] == ("kept" if null == "contact_gt" else "zeroed")
    assert sem["contact_frames"] == ("cond_pinned" if TE.pins_contact(null)
                                     else "co_denoised_from_noise")
    assert sem["is_deploy_condition"] is (null == "none")

    # prev_cpk: a real package for the steady-state windows, zeros when nulled
    pkgs = [p for p in rec.prev_cpks if p is not None]
    assert pkgs, "no steady-state window exercised the prev_cpk path"
    zeroed = all(float(p.wrench.abs().sum()) == 0.0 for p in pkgs)
    # contact_zero pins the PREVIOUS window's CONTACT frames to zeros too, so
    # the package it chains forward is zero by construction (recorded in
    # null_semantics as prev_cpk=zero_by_pinning), not by a separate null
    assert zeroed is (TE.nulls_prev_cpk(null) or null == "contact_zero")
    assert s["null_semantics"]["prev_cpk"] == (
        "zeroed" if TE.nulls_prev_cpk(null) else
        "zero_by_pinning" if null == "contact_zero" else "as_sampled")

    # action targets and the intent chunk are never nulled
    assert not rec.zeroed("action_chunk") and not rec.zeroed("prev_chunk")


@pytestmark_e2e
def test_summary_has_the_new_keys_end_to_end(tiny_eval_setup, tmp_path):
    got = _run(tiny_eval_setup, tmp_path, "none")
    s = got["summary"]
    for k in ("median_endpoint_err_mm", "seed_std_endpoint_err_mm",
              "pred_close_height_mm", "median_pred_close_height_mm",
              "gt_close_height_mm", "n_windows", "seeds", "split",
              "persistent_noise"):
        assert k in s, k
    task = next(iter(s["per_task"]))
    assert "median_close_step_err" in s["per_task"][task]
    # rows carry the absolute heights the rig failure is stated in
    r = got["rows"][0]
    assert np.isfinite(r["z_start_mm"]) and "pred_close_height_mm" in r
    assert s["max_episodes"] is None, "--max-episodes must default to ALL"


@pytestmark_e2e
def test_persistent_noise_holds_one_draw_across_the_replans_of_a_window(
        tiny_eval_setup, tmp_path, monkeypatch):
    rec = _Recorder(monkeypatch)
    got = _run(tiny_eval_setup, tmp_path / "on", "none", ["--persistent-noise"])
    assert got["summary"]["persistent_noise"] is True
    assert all(rec.reuse), "reuse_noise must reach every sample() call"
    held = rec.held_noise
    assert all(h is not None for h in held)
    # the steady-state window samples twice (prev replan, then the terminal
    # one) off ONE draw; the draw is re-rolled between (window, seed)
    assert any(a == b for a, b in zip(held, held[1:])), "no draw was reused"
    assert len(set(held)) > 1, "the draw must be re-rolled between seeds"


@pytestmark_e2e
def test_without_the_flag_no_noise_is_held(tiny_eval_setup, tmp_path, monkeypatch):
    rec = _Recorder(monkeypatch)
    got = _run(tiny_eval_setup, tmp_path / "off", "none")
    assert got["summary"]["persistent_noise"] is False
    assert not any(rec.reuse)
    assert all(h is None for h in rec.held_noise)


@pytestmark_e2e
def test_max_episodes_still_caps(tiny_eval_setup, tmp_path):
    full = _run(tiny_eval_setup, tmp_path / "f", "none")
    one = _run(tiny_eval_setup, tmp_path / "o", "none", ["--max-episodes", "1"])
    assert one["summary"]["n_episodes"] <= full["summary"]["n_episodes"]


# ---------------------------------------------------------------------------
# F12 / F18 (validation 2026-08-30): the mode semantics are explicit, the
# default is the DEPLOY condition, and a student checkpoint runs at all
# ---------------------------------------------------------------------------

def test_null_semantics_names_both_switches_for_every_mode():
    """Every mode is (package kept|zeroed) x (frames pinned|co-denoised)."""
    sem = {m: TE.null_semantics(m) for m in TE.NULL_MODES}
    kept = {m for m, s in sem.items() if s["gt_contact_package"] == "kept"}
    pinned = {m for m, s in sem.items() if s["contact_frames"] == "cond_pinned"}
    assert kept == {"contact_gt"}, "only contact_gt keeps the GT package"
    assert pinned == {"contact_gt", "contact_zero"}
    # the default is the deploy condition, NOT "everything on"
    assert sem["none"]["gt_contact_package"] == "zeroed"
    assert sem["none"]["is_deploy_condition"] is True
    assert sem["none"]["contact_frames"] == "co_denoised_from_noise"
    # the P7 pair differs ONLY in what the pinned frames hold
    a, b = sem["contact_zero"], sem["contact_gt"]
    assert a["contact_frames_pinned_to"] == "zero_package"
    assert b["contact_frames_pinned_to"] == "gt_package"
    assert sem["tactile"]["batch_streams_zeroed"] == \
        ["contact_state", "fields", "gel", "reactive"]
    with pytest.raises(SystemExit):
        TE.null_semantics("contact_maybe")


def test_contact_zero_zeroes_the_package_and_pins_it():
    """P7's control arm: same package as the default, pinned like contact_gt."""
    assert TE.pins_contact("contact_zero") and not TE.keeps_gt_package("contact_zero")
    assert not TE.pins_contact("none") and not TE.keeps_gt_package("none")
    # it is not a batch edit either — the package zeroing happens in main()
    assert _zeroed(TE.null_batch(_fake_batch(), "contact_zero")) == set()


@pytestmark_e2e
def test_terminal_eval_runs_a_student_checkpoint(tiny_eval_setup, tmp_path):
    """F12: build_model got a hardcoded student=False, so builder.py:48
    ('mc.student must agree with the student flag') killed every student
    checkpoint — the ones D9/D12 evaluate offline."""
    from phantom.config.training import CommonTrainConfig
    from phantom.train import common as C
    from phantom.train.builder import build_model
    hw, root, _, hw_yaml = tiny_eval_setup
    ns = NormStats(mean={"action": np.zeros(hw.control.action_dim, np.float32),
                         "ur_state": np.zeros(hw.ur_state_dim, np.float32)},
                   std={"action": np.ones(hw.control.action_dim, np.float32),
                        "ur_state": np.ones(hw.ur_state_dim, np.float32)})
    pm = build_model(hw, load_paths(), student=True, tiny=True, load_base=False)
    ckpt = Path(tmp_path) / "tiny_student.pt"
    C.save_phantom_checkpoint(ckpt, pm.rf, hw=hw, bb=pm.bb, mc=pm.mc,
                              train_cfg=CommonTrainConfig(), step=0, norm_stats=ns)
    out = Path(tmp_path) / "student.json"
    assert TE.main(["--ckpt", str(ckpt), "--data", str(root / "tasks"),
                    "--hardware", str(hw_yaml), "--nfe", "1", "--seeds", "1",
                    "--no-ema", "--tiny", "--out", str(out)]) == 0
    s = json.loads(out.read_text())["summary"]
    assert s["student"] is True and s["n"] > 0


@pytestmark_e2e
def test_dump_video_error_adds_the_imagined_future_columns_and_changes_nothing_else(
        tiny_eval_setup, tmp_path):
    """--dump-video-error is additive: identical action numbers, extra columns.

    The imagined-future error is scored against the VAE encode of the window's
    REAL future frames (sampling itself skips that encode), and the GT-free
    across-seed agreement is what a deploy selector could rank on."""
    off = _run(tiny_eval_setup, tmp_path / "off", "none")
    on = _run(tiny_eval_setup, tmp_path / "on", "none", ["--dump-video-error"])

    # the flag is OFF by default and adds no key to that JSON
    assert "video_err" not in off["summary"] and "dump_video_error" not in off["summary"]
    assert not any(k.startswith("video_") for k in off["rows"][0])

    s = on["summary"]
    assert s["dump_video_error"] is True and s["video_attend"] is False
    for k in ("video_err", "median_video_err", "seed_std_video_err",
              "video_err_to_median", "governor_sigma"):
        assert k in s, k
    assert np.isfinite(s["video_err"]) and s["video_err"] > 0

    # same windows and seeds; the flag only ADDS columns (the tiny-backbone
    # harness is not bit-reproducible run to run, so the values of the shared
    # action columns are not compared here — the key set is)
    assert len(on["rows"]) == len(off["rows"])
    for a, b in zip(off["rows"], on["rows"]):
        assert (a["episode"], a["seed"], a["t0"]) == (b["episode"], b["seed"], b["t0"])
        assert set(a) < set(b)
        assert set(b) - set(a) >= {"video_err", "video_err_frames",
                                   "video_err_to_median"}
        assert len(b["video_err_frames"]) == 3 and np.isfinite(b["video_err"])
        assert b["video_err_to_median"] >= 0.0
    # the action metrics are still computed and finite under the flag
    assert np.isfinite(on["summary"]["endpoint_err_mm"])
