"""Per-cell grasp/force report for one arm on one task from the grasp_events CSV.
usage: arm_task_report.py <csv> <arm label> <task substring>"""
import csv
import statistics as st
import sys

path, arm, task = sys.argv[1], sys.argv[2], sys.argv[3]
rows = [r for r in csv.DictReader(open(path)) if task in r["episode"] and r["arm"] == arm]


def pk(r):
    try:
        return float(r["pad_peak_take_n"] or r["pad_peak_n"])
    except ValueError:
        return float("nan")


print(f"{arm} on {task}: {len(rows)} takes")
for r in sorted(rows, key=lambda r: int(r["cell"])):
    print(f"  cell {int(r['cell']) - 100:2d} verdict={r['verdict'][:10]:10s} closes={r['n_closes']:>2s} "
          f"pad_peak={pk(r):5.1f} N lift={r['lift_mm']:>6s} stop={r['stop_reason']}")
placed = [r for r in rows if r["verdict"].startswith("s")]
failed = [r for r in rows if not r["verdict"].startswith("s")]
print("placed:", len(placed), "| pad peaks placed:", sorted(round(pk(r), 1) for r in placed))
if placed:
    print("placed force mean %.1f sd %.1f" % (st.mean(pk(r) for r in placed), st.pstdev(pk(r) for r in placed)))
print("failed:", len(failed), "| touched (pad>2.5 N) but failed:", sum(1 for r in failed if pk(r) > 2.5),
      "| never touched:", sum(1 for r in failed if pk(r) <= 2.5), "| no close command:", sum(1 for r in failed if r["n_closes"] == "0"))
