import csv, json, os, collections, statistics as st
import numpy as np
ROOT = os.path.expanduser("~/phantom-icra-2027/data/episodes/deploy/20260915_experiment")
rows = list(csv.DictReader(open("/tmp/final_analysis/ge_final.csv")))
ARMS = ["v6_simft2k", "stu_simft_001000", "pi05", "dp"]
TASKS = ["waffles", "Carton"]
out = {}
for r in rows:
    d = os.path.join(ROOT, r["episode"])
    m = json.load(open(os.path.join(d, "meta.json")))
    task = m.get("task")
    dur = None
    for src in ("actions.zarr", "arm_q.zarr", "gripper.zarr"):
        tsp = os.path.join(d, src, "ts")
        if os.path.isdir(tsp):
            try:
                import zarr
                z = zarr.open(os.path.join(d, src), mode="r")
                ts = np.asarray(z["ts"][:])
                if ts.size >= 2:
                    dur = float(ts[-1] - ts[0]); break
            except Exception as e:
                pass
    sj = os.path.join(d, "stop.json")
    ev, nrep, stopr = [], None, None
    if os.path.exists(sj):
        try:
            s = json.load(open(sj))
            ev = [e.get("kind") for e in (s.get("safety_events") or [])]
            nrep = s.get("n_replans")
            stopr = s.get("stopped_reason")
        except Exception: pass
    out[r["episode"]] = dict(task=task, arm=r["arm"], dur=dur, ev=ev, nrep=nrep, stop=stopr)

print("dur available: %d/%d" % (sum(1 for v in out.values() if v["dur"] is not None), len(out)))
print()
print("task,arm,n,mean_dur_s,median_dur_s,mean_replans,wrench_limit,protective_stop,control_lost,veto_retry_cap,reach_clamp,workspace_clamp,tactile_depth,other_events,stop_reasons")
for task in TASKS:
    for arm in ARMS:
        g = [v for v in out.values() if v["task"] == task and v["arm"] == arm]
        durs = [v["dur"] for v in g if v["dur"] is not None]
        reps = [v["nrep"] for v in g if isinstance(v["nrep"], (int, float))]
        evc = collections.Counter(e for v in g for e in v["ev"])
        stops = collections.Counter(v["stop"] for v in g)
        known = ("wrench_limit", "protective_stop", "control_lost", "veto_retry_cap", "reach_clamp", "workspace_clamp", "tactile_depth")
        other = {k: c for k, c in evc.items() if k not in known}
        print("%s,%s,%d,%s,%s,%s,%d,%d,%d,%d,%d,%d,%d,%s,%s" % (
            task, arm, len(g),
            "%.1f" % st.mean(durs) if durs else "",
            "%.1f" % st.median(durs) if durs else "",
            "%.1f" % st.mean(reps) if reps else "",
            evc.get("wrench_limit", 0), evc.get("protective_stop", 0), evc.get("control_lost", 0),
            evc.get("veto_retry_cap", 0), evc.get("reach_clamp", 0), evc.get("workspace_clamp", 0),
            evc.get("tactile_depth", 0), other or "{}", dict(stops)))
json.dump({k: {kk: vv for kk, vv in v.items()} for k, v in out.items()}, open("/tmp/final_analysis/dur_safety.json", "w"))
