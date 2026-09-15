import collections, csv, statistics as st, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["verdict"] not in ("", "None", "?")]
def pk(r):
    for k in ("pad_peak_take_n", "pad_peak_n"):
        try: return float(r[k])
        except (ValueError, KeyError): pass
    return float("nan")
def task(r): return "waffles" if "waffles" in r["episode"] else "Carton" if "Carton" in r["episode"] else "egg"
def yes(v): return str(v).lower() in ("true", "1", "yes")
by = collections.defaultdict(list)
for r in rows: by[(task(r), r["arm"])].append(r)
print(f"{'task':8s} {'arm':17s} {'n':>2s} {'placed':>6s} {'grasp':>5s} {'proper':>6s} {'over':>4s} {'under':>5s} {'force placed':>14s} {'force proper':>14s}")
for k in sorted(by):
    rs = by[k]
    placed = [r for r in rs if r["verdict"].startswith(("s", "c"))]
    grasp = [r for r in rs if yes(r["grasp_success"])]
    proper = [r for r in rs if yes(r["haptic_success"])]
    over = [r for r in rs if yes(r["over_grasp"]) or yes(r["crush"])]
    under = [r for r in rs if yes(r["under_grasp"])]
    fp = [pk(r) for r in placed]; fq = [pk(r) for r in proper]
    f1 = f"{st.mean(fp):.1f}±{st.pstdev(fp):.1f}" if fp else "-"
    f2 = f"{st.mean(fq):.1f}±{st.pstdev(fq):.1f}" if fq else "-"
    print(f"{k[0]:8s} {k[1]:17s} {len(rs):2d} {len(placed):6d} {len(grasp):5d} {len(proper):6d} {len(over):4d} {len(under):5d} {f1:>14s} {f2:>14s}")
    print("     under cells:", [int(r['cell'])-100 for r in under], "over cells:", [(int(r['cell'])-100, pk(r)) for r in over])
