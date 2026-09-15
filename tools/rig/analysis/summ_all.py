"""Per (task, arm) summary from the grasp_events CSV: operator placements, sensor classes, pad force on placements."""
import collections
import csv
import statistics as st
import sys

rows = list(csv.DictReader(open(sys.argv[1])))
print("columns:", list(rows[0].keys()))
cls_col = next((c for c in ("cls", "class", "outcome", "sensor_class") if c in rows[0]), None)


def pk(r):
    for k in ("pad_peak_take_n", "pad_peak_n"):
        try:
            return float(r[k])
        except (ValueError, KeyError):
            pass
    return float("nan")


def task_of(r):
    return "waffles" if "waffles" in r["episode"] else "Carton" if "Carton" in r["episode"] else "egg" if "egg" in r["episode"] else "?"


by = collections.defaultdict(list)
for r in rows:
    by[(task_of(r), r["arm"])].append(r)
for (task, arm) in sorted(by):
    rs = by[(task, arm)]
    placed = [r for r in rs if r["verdict"].startswith("s")]
    nov = [r for r in rs if not r["verdict"] or r["verdict"] in ("None", "?")]
    f = [pk(r) for r in placed]
    over = sum(1 for x in f if x > 21.3)
    classes = collections.Counter(r[cls_col] for r in rs) if cls_col else {}
    cells = sorted(int(r["cell"]) - 100 for r in rs)
    print(f"\n{task:7s} {arm:16s} n={len(rs):2d} cells={cells[0]}-{cells[-1]} placed={len(placed):2d} no-verdict={len(nov)}")
    if f:
        print(f"   pad on placements: mean {st.mean(f):.1f} sd {st.pstdev(f):.1f} max {max(f):.1f} over21.3N={over}")
    print("   classes:", dict(classes))
    fails = [r for r in rs if not r["verdict"].startswith("s")]
    print(f"   fails={len(fails)}: touched(>2.5N)={sum(1 for r in fails if pk(r) > 2.5)} never_touched={sum(1 for r in fails if pk(r) <= 2.5)} no_close_cmd={sum(1 for r in fails if r.get('n_closes') == '0')}")
    stops = collections.Counter(r.get("stop_reason", "") for r in rs)
    print("   stops:", dict(stops))
    for r in nov:
        print("   NO VERDICT:", r["episode"], "cell", int(r["cell"]) - 100, "pad", pk(r), "stop", r.get("stop_reason"))
