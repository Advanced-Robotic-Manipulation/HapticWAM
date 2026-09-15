import csv, json, os, collections, math, statistics as st

CSV = "/tmp/final_analysis/ge_final.csv"
ROOT = os.path.expanduser("~/phantom-icra-2027/data/episodes/deploy/20260915_experiment")
CRUSH = 21.3
ARMS = ["v6_simft2k", "stu_simft_001000", "pi05", "dp"]
TASKS = ["waffles", "Carton"]

rows = list(csv.DictReader(open(CSV)))
def f(r, k):
    v = r.get(k, "")
    try: return float(v)
    except: return None
def b(r, k):
    return str(r.get(k, "")).strip().lower() in ("1", "true", "yes")

# attach task + duration + evt tags + crushed tag from meta.json
for r in rows:
    mp = os.path.join(ROOT, r["episode"], "meta.json")
    m = json.load(open(mp))
    r["_task"] = m.get("task")
    tags = m.get("tags") or []
    r["_evts"] = [t[4:] for t in tags if t.startswith("evt:")]
    r["_crushed_tag"] = "crushed" in tags
    dur = None
    for k in ("duration_s", "duration", "t_end_s", "elapsed_s"):
        if isinstance(m.get(k), (int, float)): dur = float(m[k]); break
    if dur is None:
        s, e = m.get("t_start"), m.get("t_end")
        if isinstance(s, (int, float)) and isinstance(e, (int, float)): dur = float(e) - float(s)
    if dur is None:
        s, e = m.get("started_at"), m.get("ended_at")
        if isinstance(s, (int, float)) and isinstance(e, (int, float)): dur = float(e) - float(s)
    r["_dur"] = dur
    r["_notes_meta"] = (m.get("notes") or "")
    r["_success_meta"] = m.get("success")

def ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals: return "n/a"
    if len(vals) == 1: return "%.1f (n=1)" % vals[0]
    return "%.1f±%.1f [%.1f–%.1f]" % (st.mean(vals), st.stdev(vals), min(vals), max(vals))

print("###META_KEYS", sorted(json.load(open(os.path.join(ROOT, rows[0]["episode"], "meta.json"))).keys()))
print("###DUR_AVAIL", sum(1 for r in rows if r["_dur"] is not None), "/", len(rows))
print()

print("### PER TASK/ARM FULL")
for task in TASKS:
    for arm in ARMS:
        g = [r for r in rows if r["_task"] == task and r["arm"] == arm]
        placed_op = [r for r in g if b(r, "operator_placed") or r["_crushed_tag"]]
        grasp = [r for r in g if b(r, "grasp_success")]
        proper = [r for r in g if b(r, "haptic_success")]
        over = [(int(r["cell"]) - 100 if int(r["cell"]) > 100 else int(r["cell"]), f(r, "pad_peak_pinch_n")) for r in g if b(r, "over_grasp")]
        under = sorted(int(r["cell"]) - 100 if int(r["cell"]) > 100 else int(r["cell"]) for r in g if b(r, "under_grasp"))
        cls = collections.Counter(r["class"] for r in g)
        stops = collections.Counter(r["stop_reason"] for r in g)
        evts = collections.Counter(e for r in g for e in r["_evts"])
        pinch_placed = [f(r, "pad_peak_pinch_n") for r in g if r["class"] in ("placed_clean", "placed_crushed")]
        pinch_proper = [f(r, "pad_peak_pinch_n") for r in g if b(r, "haptic_success")]
        reps = [f(r, "n_replans") for r in g]
        durs = [r["_dur"] for r in g]
        fails = [r for r in g if not (b(r, "operator_placed") or r["_crushed_tag"])]
        touched = sum(1 for r in fails if (f(r, "pad_peak_take_n") or 0) > 2.5)
        noclose = sum(1 for r in fails if (f(r, "n_closes") or 0) == 0)
        print("== %s / %s  n=%d" % (task, arm, len(g)))
        print("   op_placed(incl crushed)=%d  grasp=%d  proper=%d  over=%d  under=%d" %
              (len(placed_op), len(grasp), len(proper), len(over), len(under)))
        print("   over cells(N): %s" % over)
        print("   under cells: %s" % under)
        print("   never_reached=%d closed_on_air=%d" % (cls.get("never_reached", 0), cls.get("closed_on_air", 0)))
        print("   classes: %s" % dict(cls))
        print("   pinch placements: %s   pinch proper: %s" % (ms(pinch_placed), ms(pinch_proper)))
        print("   fails=%d  touched=%d  never_touched=%d  no_close_cmd=%d" % (len(fails), touched, len(fails) - touched, noclose))
        print("   stops: %s" % dict(stops))
        print("   evts: %s" % dict(evts))
        print("   replans mean=%.2f  dur mean=%s s" % (st.mean([x for x in reps if x is not None]) if any(x is not None for x in reps) else -1,
              ("%.1f" % st.mean([d for d in durs if d is not None])) if any(d is not None for d in durs) else "n/a"))
        print()

print("### DISCREPANCIES (operator vs sensor)")
for r in rows:
    op = b(r, "operator_placed") or r["_crushed_tag"]
    sens_placed = r["class"] in ("placed_clean", "placed_crushed")
    if op != sens_placed or (r["_crushed_tag"] != b(r, "crush")):
        print("  %-34s %-10s %-17s cell %-3s op_placed=%s op_crushed=%s class=%-15s crush_sensor=%s pinch=%.1f N  notes=%r" % (
            r["episode"], r["_task"], r["arm"], r["cell"], op, r["_crushed_tag"], r["class"], b(r, "crush"),
            f(r, "pad_peak_pinch_n") or -1, r["_notes_meta"][:40]))
print()

print("### TEACHER WAFFLES cells 1-10 vs 11-20")
g = [r for r in rows if r["_task"] == "waffles" and r["arm"] == "v6_simft2k"]
for lo, hi, lbl in ((1, 10, "cells 1-10"), (11, 20, "cells 11-20")):
    h = [r for r in g if lo <= (int(r["cell"]) - 100 if int(r["cell"]) > 100 else int(r["cell"])) <= hi]
    pl = sum(1 for r in h if b(r, "operator_placed") or r["_crushed_tag"])
    gr = sum(1 for r in h if b(r, "grasp_success"))
    pr = sum(1 for r in h if b(r, "haptic_success"))
    ov = sum(1 for r in h if b(r, "over_grasp"))
    un = sum(1 for r in h if b(r, "under_grasp"))
    pin = [f(r, "pad_peak_pinch_n") for r in h if r["class"] in ("placed_clean", "placed_crushed")]
    print("  %s n=%d placed=%d grasp=%d proper=%d over=%d under=%d pinch_placed=%s" % (lbl, len(h), pl, gr, pr, ov, un, ms(pin)))
print()

print("### SUMMARY CSV")
w = csv.writer(open("/tmp/final_analysis/summary_by_arm.csv", "w", newline=""))
w.writerow(["task", "arm", "n", "operator_placed", "grasp_success", "proper_haptic", "over_grasp", "under_grasp",
            "never_reached", "closed_on_air", "pinch_placed_mean_n", "pinch_placed_sd_n", "pinch_placed_min_n",
            "pinch_placed_max_n", "pinch_proper_mean_n", "pinch_proper_sd_n", "fails", "fail_touched",
            "fail_never_touched", "fail_no_close_cmd", "safety_stop", "protective_stop", "control_lost",
            "veto_retry_cap", "operator_stop", "mean_replans", "mean_duration_s"])
for task in TASKS:
    for arm in ARMS:
        g = [r for r in rows if r["_task"] == task and r["arm"] == arm]
        cls = collections.Counter(r["class"] for r in g)
        stops = collections.Counter(r["stop_reason"] for r in g)
        pp = [f(r, "pad_peak_pinch_n") for r in g if r["class"] in ("placed_clean", "placed_crushed")]
        pr = [f(r, "pad_peak_pinch_n") for r in g if b(r, "haptic_success")]
        pp = [x for x in pp if x is not None]; pr = [x for x in pr if x is not None]
        fails = [r for r in g if not (b(r, "operator_placed") or r["_crushed_tag"])]
        touched = sum(1 for r in fails if (f(r, "pad_peak_take_n") or 0) > 2.5)
        reps = [f(r, "n_replans") for r in g if f(r, "n_replans") is not None]
        durs = [r["_dur"] for r in g if r["_dur"] is not None]
        w.writerow([task, arm, len(g),
                    sum(1 for r in g if b(r, "operator_placed") or r["_crushed_tag"]),
                    sum(1 for r in g if b(r, "grasp_success")),
                    sum(1 for r in g if b(r, "haptic_success")),
                    sum(1 for r in g if b(r, "over_grasp")),
                    sum(1 for r in g if b(r, "under_grasp")),
                    cls.get("never_reached", 0), cls.get("closed_on_air", 0),
                    round(st.mean(pp), 2) if pp else "", round(st.stdev(pp), 2) if len(pp) > 1 else "",
                    round(min(pp), 2) if pp else "", round(max(pp), 2) if pp else "",
                    round(st.mean(pr), 2) if pr else "", round(st.stdev(pr), 2) if len(pr) > 1 else "",
                    len(fails), touched, len(fails) - touched,
                    sum(1 for r in fails if (f(r, "n_closes") or 0) == 0),
                    stops.get("safety_stop", 0), stops.get("protective_stop", 0), stops.get("control_lost", 0),
                    stops.get("veto_retry_cap", 0), stops.get("operator_stop", 0),
                    round(st.mean(reps), 2) if reps else "", round(st.mean(durs), 1) if durs else ""])
print("wrote /tmp/final_analysis/summary_by_arm.csv")
