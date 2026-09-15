"""Inventory of the experiment folder: per task and arm, counts and cell completeness."""
import collections
import glob
import json
import os

from phantom.eval import stats as S

D = os.path.expanduser("~/phantom-icra-2027/data/episodes/deploy/20260915_experiment")
c = collections.defaultdict(lambda: {"n": 0, "placed": 0, "failed": 0, "nov": 0, "cells": set()})
carton = []
for e in sorted(glob.glob(D + "/ep_*")):
    m = json.load(open(e + "/meta.json"))
    lab = [t[6:] for t in m["tags"] if t.startswith("label:")]
    lab = lab[0] if lab else "?"
    k = (m.get("task"), lab)
    c[k]["n"] += 1
    c[k]["cells"].add(S.episode_seed(m) - 100)
    if m.get("success") is True:
        c[k]["placed"] += 1
    elif m.get("success") is False:
        c[k]["failed"] += 1
    else:
        c[k]["nov"] += 1
    if m.get("task") == "Carton":
        carton.append((os.path.basename(e), m["status"], m.get("success"), S.episode_seed(m) - 100))
for k in sorted(c, key=str):
    v = c[k]
    complete = sorted(v["cells"]) == list(range(1, 21))
    print(f"{str(k[0]):8s} {k[1]:16s} n={v['n']:2d} placed={v['placed']:2d} failed={v['failed']:2d} no-verdict={v['nov']} cells1-20={complete}")
print("Carton takes present:", carton)
