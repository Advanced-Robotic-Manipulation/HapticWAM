import csv, json, os
ROOT = os.path.expanduser("~/phantom-icra-2027/data/episodes/deploy/20260915_experiment")
rows = list(csv.DictReader(open("/tmp/final_analysis/ge_final.csv")))
CRUSH = 21.3
def bo(r, k): return str(r.get(k, "")).strip().lower() in ("1", "true", "yes")
def fl(r, k):
    try: return float(r[k])
    except: return None
print("%-34s %-8s %-17s %-5s %-24s %-16s %-7s %s" % ("episode", "task", "arm", "cell", "operator", "sensor class", "pinch", "kind of disagreement"))
n = 0
for r in rows:
    m = json.load(open(os.path.join(ROOT, r["episode"], "meta.json")))
    tags = m.get("tags") or []
    task = m.get("task")
    notes = (m.get("notes") or "").replace("operator: ", "")
    op_crush_tag = "crushed" in tags
    op_force = bo(r, "op_force_flag") or op_crush_tag
    op_placed = bo(r, "operator_placed") or (op_crush_tag and m.get("success") is True)
    sens_placed = r["class"] in ("placed_clean", "placed_crushed")
    sens_crush = bo(r, "crush")
    pinch = fl(r, "pad_peak_pinch_n")
    kinds = []
    if op_placed and not sens_placed: kinds.append("operator placed, sensors did not")
    if sens_placed and not op_placed: kinds.append("sensors placed, operator did not")
    if op_force and not sens_crush: kinds.append("operator flagged force, pinch below %.1f N" % CRUSH)
    if sens_crush and not op_force: kinds.append("sensors crush, operator flagged nothing")
    if kinds:
        n += 1
        cell = int(r["cell"]) - 100 if int(r["cell"]) > 100 else int(r["cell"])
        print("%-34s %-8s %-17s %-5d %-24s %-16s %-7s %s" % (
            r["episode"], task, r["arm"], cell, notes[:24], r["class"],
            "%.1f" % pinch if pinch is not None else "n/a", "; ".join(kinds)))
print("\ntotal disagreeing takes: %d / %d" % (n, len(rows)))
