"""Tactile prediction error: the predicted contact package is recorded per
replan (planner_cpk.npz + `cpk_pred` in the trace) and scored offline
against the pads' later measurements next to a persistence baseline.

Synthetic zarr episodes (real EpisodeWriter, real fields_ds streams) with a
contact onset mid-episode; predictions are built FROM the observed package
so the expected scores are exact: a perfect predictor scores skill 1, the
persistence predictor scores skill 0, a wrong one scores worse.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from phantom.config.model import EVENT_IDX, N_EVENTS
from phantom.data.episode_store import EpisodeReader, EpisodeWriter
from phantom.data.schema import EpisodeMeta, NormStats, tactile_stream
from phantom.deploy.cpk_trace import (CPK_TRACE_FILE, CpkTraceLog, load_cpk_trace,
                                      summarize_package)
from phantom.deploy.planner import PlannerLoop
from phantom.eval import tactile_prediction as TP
from phantom.eval.metrics import trial_metrics
from phantom.inference.policy import ObsSnapshot, Plan
from phantom.model.ace.packing import ContactPackage
from phantom_test_utils import make_small_hw

T0 = 1000.0
DUR = 6.0
CONTACT_T = 3.0
LATENT_DT = 0.25
TC = 3
# replan (host) times: four scoreable, the last one's horizon leaves the recording
REPLAN_T = [T0 + 1.0, T0 + 2.0, T0 + 2.9, T0 + 3.5, T0 + 5.6]


@pytest.fixture(scope="module")
def hw():
    return make_small_hw()


def _frame(hw, t_rel: float, contact_t: float) -> np.ndarray:
    """One field_ds frame: a depth+force blob after the onset whose force
    keeps growing (so d_fz != 0 during contact and persistence is wrong)."""
    h, w = hw.recording.field_ds.h, hw.recording.field_ds.w
    f = np.zeros((h, w, hw.tactile.field_ch), dtype=np.float16)
    if t_rel >= contact_t:
        amp = 1.0 + 0.5 * (t_rel - contact_t)
        f[4:12, 4:14, 2] = 1.0          # |depth| >> tau_contact_depth
        f[4:12, 4:14, 7] = amp          # dist-force z
    return f


def write_episode(path, hw, *, contact_t=CONTACT_T, wrench=True):
    w = EpisodeWriter(path, hw, EpisodeMeta(task="waffles"))
    f_hz = hw.recording.field_ds_rate_hz
    f_ts = T0 + np.arange(0.0, DUR, 1.0 / f_hz)
    frames = np.stack([_frame(hw, t - T0, contact_t) for t in f_ts])
    for s in hw.tactile.sensors:
        w.append(tactile_stream(s.name, "fields_ds"), f_ts, frames)
        if wrench:
            w_ts = T0 + np.arange(0.0, DUR, 1.0 / 8.0)
            wr = np.zeros((len(w_ts), 6), dtype=np.float32)
            wr[:, 2] = np.where(w_ts - T0 >= contact_t,
                                2.0 * (1.0 + 0.5 * (w_ts - T0 - contact_t)), 0.0)
            w.append(tactile_stream(s.name, "wrench"), w_ts, wr)
    w.finalize(success=True)
    return path


def _package(obs: dict, *, persistence: bool = False, wrong: bool = False) -> ContactPackage:
    """A (1, Tc, ...) ContactPackage from an observed package dict."""
    Tc, F = obs["d_fz"].shape[:2]
    if persistence:
        d_fz = np.zeros_like(obs["d_fz"])
        d_disp = np.zeros_like(obs["d_disp"])
        mask = np.repeat(obs["mask0"][None], Tc, 0)
        cop = np.repeat(obs["cop0"][None], Tc, 0)
        slip = np.repeat(obs["slip0"][None], Tc, 0)
        ev = np.full(Tc, EVENT_IDX["hold"] if obs["contact0"] else EVENT_IDX["none"])
        wrench = np.repeat(np.nan_to_num(obs["wrench0"])[None], Tc, 0)
    elif wrong:
        d_fz = obs["d_fz"] + 1.0
        d_disp = obs["d_disp"]
        mask = 1.0 - obs["mask"]
        cop = np.nan_to_num(obs["cop"]) + 0.5
        slip = obs["slip"] + 1.0
        ev = (obs["event"] + 1) % N_EVENTS
        wrench = np.nan_to_num(obs["wrench"]) + 1.0
    else:
        d_fz, d_disp, mask, cop, slip, ev = (obs["d_fz"], obs["d_disp"], obs["mask"],
                                             obs["cop"], obs["slip"], obs["event"])
        wrench = np.nan_to_num(obs["wrench"])
    event = np.eye(N_EVENTS, dtype=np.float32)[np.asarray(ev)]
    t = lambda a: torch.from_numpy(np.ascontiguousarray(np.asarray(a, np.float32)))[None]
    return ContactPackage(
        event=t(event), d_disp=t(np.moveaxis(d_disp, -1, 2)), d_fz=t(d_fz), mask=t(mask),
        cop=t(cop), slip=t(slip), wrench=t(wrench), wrist=t(np.zeros((Tc, 6), np.float32)))


def _record(ep, hw, kind: str = "perfect", times=REPLAN_T, offset: float = 0.0) -> int:
    """Write planner_cpk.npz for `ep` with predictions derived from the
    recording itself (`perfect` / `persistence` / `wrong`). `times` are HOST
    times (what the planner stamps); the observation lives at host + offset."""
    streams = TP._Streams(EpisodeReader(ep), hw)
    log = CpkTraceLog(hw)
    for i, t0 in enumerate(times):
        obs = TP.observed_package(streams, hw, t0 + offset, LATENT_DT, TC)
        if obs is None:                                  # beyond the recording:
            obs = TP.observed_package(streams, hw, T0 + 1.0, LATENT_DT, TC)   # any shape
        pkg = _package(obs, persistence=kind == "persistence", wrong=kind == "wrong")
        log.append(t_host=t0, latency_s=0.5, trace_index=i, cpk=pkg,
                   norm=NormStats.identity(), latent_dt=LATENT_DT)
    log.save(ep / CPK_TRACE_FILE)
    return len(log)


# ---------------------------------------------------------------------------
# observed package mirrors the training target
# ---------------------------------------------------------------------------

def test_observed_package_geometry_and_events(tmp_path, hw):
    ep = write_episode(tmp_path / "ep", hw)
    streams = TP._Streams(EpisodeReader(ep), hw)
    cph, cpw = hw.cpk_shape
    F = hw.n_fingers
    obs = TP.observed_package(streams, hw, T0 + 2.9, LATENT_DT, TC)
    assert obs["d_fz"].shape == (TC, F, cph, cpw)
    assert obs["mask"].shape == (TC, F, cph, cpw)
    # u = 2.9, 3.15, 3.4, 3.65: no contact at u0, contact from u1 => onset, hold, hold
    assert obs["contact0"] is False
    assert obs["event"].tolist() == [EVENT_IDX["onset"], EVENT_IDX["hold"], EVENT_IDX["hold"]]
    assert obs["mask"][0].mean() > 0 and obs["mask0"].mean() == 0
    # force grows during contact: d_fz positive at steps 2, 3 (inside contact)
    assert obs["d_fz"][1].max() > 0 and obs["d_fz"][2].max() > 0
    assert np.isfinite(obs["wrench"]).all()
    # a horizon that leaves the recording is not scoreable
    assert TP.observed_package(streams, hw, T0 + 5.6, LATENT_DT, TC) is None


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def test_perfect_prediction_scores_skill_one(tmp_path, hw):
    ep = write_episode(tmp_path / "ep", hw)
    n = _record(ep, hw, "perfect")
    r = TP.episode_tactile_prediction(ep, hw)
    assert r is not None and r.n_replans == n == 5
    assert r.n_scored == 4                      # the last replan's horizon is unrecorded
    s = r.summary()
    assert s["tpe_dfz_rmse"] == pytest.approx(0.0)
    assert s["tpe_dfz_rmse_persist"] > 0        # persistence cannot see the onset / ramp
    assert s["tpe_dfz_skill"] == pytest.approx(1.0)
    assert s["tpe_mask_iou"] == pytest.approx(1.0)
    assert s["tpe_event_acc"] == pytest.approx(1.0)
    assert s["tpe_cop_err"] == pytest.approx(0.0)
    assert s["tpe_wrench_rmse"] == pytest.approx(0.0)
    assert s["tpe_wrench_rmse_persist"] > 0
    # horizon-resolved keys exist for every step ahead
    assert all(f"tpe_dfz_rmse_s{k}" in s for k in range(1, TC + 1))
    assert 0 < s["tpe_contact_step_frac"] < 1


def test_persistence_prediction_scores_skill_zero(tmp_path, hw):
    ep = write_episode(tmp_path / "ep", hw)
    _record(ep, hw, "persistence")
    s = TP.episode_tactile_prediction(ep, hw).summary()
    assert s["tpe_dfz_rmse"] == pytest.approx(s["tpe_dfz_rmse_persist"])
    assert s["tpe_dfz_skill"] == pytest.approx(0.0, abs=1e-6)
    assert s["tpe_mask_iou"] == pytest.approx(s["tpe_mask_iou_persist"])
    assert s["tpe_event_acc"] == pytest.approx(s["tpe_event_acc_persist"])
    assert s["tpe_wrench_rmse"] == pytest.approx(s["tpe_wrench_rmse_persist"])


def test_wrong_prediction_is_worse_than_persistence(tmp_path, hw):
    ep = write_episode(tmp_path / "ep", hw)
    _record(ep, hw, "wrong")
    s = TP.episode_tactile_prediction(ep, hw).summary()
    assert s["tpe_dfz_skill"] < 0
    assert s["tpe_mask_iou"] < s["tpe_mask_iou_persist"]
    assert s["tpe_event_acc"] == pytest.approx(0.0)


def test_episode_without_package_is_none_and_trial_metrics_degrade(tmp_path, hw):
    ep = write_episode(tmp_path / "ep", hw)
    assert TP.episode_tactile_prediction(ep, hw) is None
    assert not any(k.startswith("tpe_") for k in trial_metrics(ep, hw, tau_obj=1.0))
    _record(ep, hw, "perfect")
    m = trial_metrics(ep, hw, tau_obj=1.0)
    assert m["tpe_dfz_skill"] == pytest.approx(1.0)
    assert m["tpe_n_replans_scored"] == 4


def test_model_space_package_is_reported_not_scored(tmp_path, hw):
    """No norm stats at deploy => the npz holds model-space values; scoring
    would compare apples to oranges, so it is refused with a reason."""
    ep = write_episode(tmp_path / "ep", hw)
    streams = TP._Streams(EpisodeReader(ep), hw)
    obs = TP.observed_package(streams, hw, T0 + 1.0, LATENT_DT, TC)
    log = CpkTraceLog(hw)
    log.append(t_host=T0 + 1.0, latency_s=0.5, trace_index=0, cpk=_package(obs),
               norm=None, latent_dt=LATENT_DT)
    log.save(ep / CPK_TRACE_FILE)
    r = TP.episode_tactile_prediction(ep, hw)
    assert r.n_scored == 0 and "model space" in r.skipped_reason
    assert r.summary()["tpe_n_replans_scored"] == 0


def test_host_clock_offset_is_applied(tmp_path, hw):
    """The npz stamps host time; streams are master time (issue #10)."""
    ep = write_episode(tmp_path / "ep", hw)
    offset = 500.0
    meta = json.loads((ep / "meta.json").read_text())
    meta["clock_calibration"] = {"offset": offset, "n_samples": 1, "spread_s": 0.0}
    (ep / "meta.json").write_text(json.dumps(meta))
    _record(ep, hw, "perfect", times=[t - offset for t in REPLAN_T], offset=offset)
    s = TP.episode_tactile_prediction(ep, hw).summary()
    assert s["tpe_n_replans_scored"] == 4 and s["tpe_dfz_skill"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# across-episode helpers
# ---------------------------------------------------------------------------

def test_pooling_horizon_and_paired_helpers(tmp_path, hw):
    a = write_episode(tmp_path / "a", hw); _record(a, hw, "perfect")
    b = write_episode(tmp_path / "b", hw); _record(b, hw, "persistence")
    ra, rb = (TP.episode_tactile_prediction(p, hw) for p in (a, b))
    pooled = TP.pool_steps([ra, rb])
    assert pooled["dfz_se"].shape == (8, TC)
    h = TP.horizon_curves([ra, rb])
    assert h["steps_ahead"] == [1, 2, 3] and len(h["dfz_rmse"]) == TC
    assert TP.sign_test_p([1, 1, 1, 1, 1, 1]) == pytest.approx(2 / 64)
    assert np.isnan(TP.sign_test_p([0.0, 0.0]))
    pc = TP.paired_compare([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], n_boot=200)
    assert pc["n_pairs"] == 3 and pc["mean_diff"] == pytest.approx(1.0)
    assert pc["n_a_better"] == 3 and pc["sign_p"] == pytest.approx(0.25)
    pe = TP.paired_compare([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], higher_is_better=False, n_boot=200)
    assert pe["n_a_better"] == 0 and pe["n_b_better"] == 3      # an error metric: lower wins
    lo, hi = TP.bootstrap_ci([0.2, 0.4, 0.6, 0.8], n_boot=500)
    assert 0.2 <= lo <= 0.5 <= hi <= 0.8


def test_aggregate_report_carries_the_tpe_table(tmp_path, hw):
    """run_eval's aggregate: per-system pooled TPE + CI, and the report table."""
    import csv
    from phantom.eval.aggregate import aggregate, write_report
    from phantom.eval.trial_runner import LEDGER_FIELDS
    eps = []
    for system, kind in (("teacher", "perfect"), ("teacher", "perfect"),
                         ("vision_only", "persistence")):
        ep = write_episode(tmp_path / f"ep_{system}_{len(eps)}", hw)
        _record(ep, hw, kind)
        eps.append((system, ep))
    ledger = tmp_path / "ledger.csv"
    with open(ledger, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
        w.writeheader()
        for i, (system, ep) in enumerate(eps):
            w.writerow({"timestamp": "t", "campaign": "c", "task": "waffles",
                        "system": system, "seed": str(i % 2), "trial": str(i),
                        "occlusion": "none", "episode_path": str(ep), "success": "True",
                        "damage": "False", "stopped_reason": "", "n_replans": "5",
                        "notes": ""})
    res = aggregate(ledger, hw, compute_episode_metrics=True)
    tpe = res["tactile_prediction"]
    assert set(tpe) == {"teacher", "vision_only"}
    assert tpe["teacher"]["summary"]["tpe_dfz_skill"] == pytest.approx(1.0)
    assert tpe["vision_only"]["summary"]["tpe_dfz_skill"] == pytest.approx(0.0, abs=1e-6)
    assert tpe["teacher"]["horizon"]["steps_ahead"] == [1, 2, 3]
    assert res["episode_metrics"]["teacher"]["tpe_dfz_skill"] == pytest.approx(1.0)
    report = write_report(res, tmp_path / "report")
    text = report.read_text()
    assert "Tactile prediction error" in text and "| teacher | 2 | 8/10 |" in text
    json.loads((tmp_path / "report" / "tactile_prediction.json").read_text())


# ---------------------------------------------------------------------------
# recording side: the planner loop keeps the package and stamps the trace
# ---------------------------------------------------------------------------

def _cpk(B=1, Tc=3, F=2, cph=3, cpw=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.rand(*s, generator=g)
    return ContactPackage(
        event=torch.softmax(r(B, Tc, 5), -1), d_disp=r(B, Tc, F, 3, cph, cpw),
        d_fz=r(B, Tc, F, cph, cpw), mask=r(B, Tc, F, cph, cpw), cop=r(B, Tc, F, 2) * 2 - 1,
        slip=r(B, Tc, F), wrench=r(B, Tc, F, 6), wrist=r(B, Tc, 6))


class _Ex:
    def __init__(self):
        self.stopped_reason = None
        self.submitted = []

    def last_cmd(self):
        return None

    def request_stop(self, reason):
        self.stopped_reason = self.stopped_reason or reason

    def submit(self, plan):
        self.submitted.append(np.array(plan.actions, copy=True))
        return True


class _Snaps:
    def __init__(self, hw):
        self.hw = hw

    def build(self):
        hw = self.hw
        return ObsSnapshot(t=time.perf_counter(), rgb=np.zeros((4, 4, 3), np.uint8),
                           wrist_window=np.zeros((hw.wrist_ft.window_len, 6), np.float32),
                           ur_state=np.zeros(2 * hw.arm.dof + 14, np.float32))


class _Pol:
    def __init__(self, hw, with_cpk=True):
        self.hw, self.with_cpk, self.n = hw, with_cpk, 0
        self.norm = NormStats.identity()
        self.latent_dt = LATENT_DT

    def replan(self, snap, prev_plan, tcp_pose):
        H, A = self.hw.control.chunk_horizon, self.hw.control.action_dim
        self.n += 1
        cpk = _cpk(seed=self.n) if self.with_cpk else None
        if cpk is not None:
            cpk.cop[0, 0, 0] = float("nan")             # an undefined CoP must survive JSON
        return Plan(t_created=float(self.n), t0_pose=np.zeros(6), actions=np.zeros((H, A)),
                    action_times=np.arange(H) / 10.0, sigma=np.zeros(3), gate=0.0,
                    p_evt=np.array([1.0, 0, 0, 0, 0]), cpk=cpk, latency_s=0.9)


def test_planner_records_package_and_trace_digest(tmp_path, hw):
    log = CpkTraceLog(hw)
    loop = PlannerLoop(hw, _Pol(hw), _Snaps(hw), _Ex(), cpk_log=log)
    loop.run(max_replans=2)
    assert len(loop.trace) == 2 and len(log) == 2
    for i, row in enumerate(loop.trace):
        assert row["cpk_index"] == i
        d = row["cpk_pred"]
        assert set(d) == {"mask_frac", "dfz_mean", "dfz_absmax", "slip", "cop", "event", "p_event"}
        assert len(d["event"]) == 3 and len(d["mask_frac"]) == 3 and len(d["mask_frac"][0]) == 2
    assert loop.trace[0]["cpk_pred"]["cop"][0][0][0] is None      # NaN -> null
    json.dumps(loop.trace)                                          # trace stays JSON-clean
    p = log.save(tmp_path / CPK_TRACE_FILE)
    z = load_cpk_trace(tmp_path)
    assert z["d_fz"].shape == (2, 3, 2, 3, 4) and z["d_disp"].shape == (2, 3, 2, 3, 4, 3)
    assert z["trace_index"].tolist() == [0, 1] and float(z["latent_dt"]) == LATENT_DT
    assert int(z["denormalized"]) == 1 and list(z["sensors"]) == [s.name for s in hw.tactile.sensors]
    assert p.exists()


def test_planner_without_package_or_log_leaves_the_trace_untouched(hw):
    loop = PlannerLoop(hw, _Pol(hw, with_cpk=False), _Snaps(hw), _Ex(), cpk_log=CpkTraceLog(hw))
    loop.run(max_replans=2)
    assert all("cpk_pred" not in r and "cpk_index" not in r for r in loop.trace)
    assert len(loop.cpk_log) == 0
    loop = PlannerLoop(hw, _Pol(hw), _Snaps(hw), _Ex())             # default: no log
    loop.run(max_replans=1)
    assert "cpk_pred" not in loop.trace[0]


def test_denormalization_uses_the_training_keys(hw):
    """The npz must hold recorder units: the inverse of WindowSampler's
    `cpk_d_fz` / `cpk_d_disp` / `wrench` / `wrist_ft` normalization."""
    norm = NormStats(mean={"cpk_d_fz": np.array([1.0], np.float32),
                           "cpk_d_disp": np.zeros(3, np.float32),
                           "wrench": np.zeros(6, np.float32),
                           "wrist_ft": np.zeros(6, np.float32)},
                     std={"cpk_d_fz": np.array([2.0], np.float32),
                          "cpk_d_disp": np.full(3, 4.0, np.float32),
                          "wrench": np.full(6, 3.0, np.float32),
                          "wrist_ft": np.full(6, 5.0, np.float32)})
    cpk = _cpk()
    log = CpkTraceLog(hw)
    log.append(t_host=0.0, latency_s=0.1, trace_index=0, cpk=cpk, norm=norm, latent_dt=LATENT_DT)
    z = log.arrays()
    np.testing.assert_allclose(z["d_fz"][0], cpk.d_fz[0].numpy() * 2.0 + 1.0, rtol=1e-5)
    # package layout (Tc, F, 3, cph, cpw) -> recorder layout (Tc, F, cph, cpw, 3)
    np.testing.assert_allclose(z["d_disp"][0], np.moveaxis(cpk.d_disp[0].numpy(), 2, -1) * 4.0,
                               rtol=1e-5)
    np.testing.assert_allclose(z["wrench"][0], cpk.wrench[0].numpy() * 3.0, rtol=1e-5)
    np.testing.assert_allclose(z["wrist"][0], cpk.wrist[0].numpy() * 5.0, rtol=1e-5)
    np.testing.assert_allclose(z["mask"][0], cpk.mask[0].numpy(), rtol=1e-6)   # never scaled
    s = summarize_package({k: z[k][0] for k in ("d_fz", "mask", "cop", "slip", "event")})
    assert s["dfz_mean"][0][0] == pytest.approx(float(z["d_fz"][0, 0, 0].mean()), abs=1e-4)
