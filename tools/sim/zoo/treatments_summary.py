#!/usr/bin/env python3
"""Paired summary of the placement-fix treatments (D1/D2/D3) against the Stage C v6 baseline.

Trials that ran after the 2026-09-12 reboot (study E3) are read from their
trial_result.json; trials that finished before the reboot survive only as
outcome labels (E1_E2_partial_after_reboot.json) and are marked 'label_only'.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

STAGE = {"no_grab": 0, "grab": 1, "pick": 2, "in_box_gripped": 3, "placed": 4}


def sign_test(wins, losses):
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    return round(min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n), 4)


def key(start):
    return start.replace("ep_waffles_", "").replace("_open", "")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--docs", type=Path, default=Path("docs/results/sim_zoo_20260912"))
    ap.add_argument("--e3-raw", type=Path, default=Path("artifacts/isaac_waffles/sim_zoo_20260912/raw/E3"))
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    base = {key(r["start"]): r for r in json.loads((args.docs / "report/C/trials.json").read_text())
            if r["model"] == "v6" and r.get("stage_name")}
    part = json.loads((args.docs / "E1_E2_partial_after_reboot.json").read_text())
    arms = {"D1": {}, "D2": {}, "D3": {}}
    for k, v in part["E1_D1_labels"].items():
        arms["D1"][k] = dict(stage_name=v.split(" ")[0], source="label_only", note=v, stop=part.get("E1_D1_stop_reasons_seen", {}).get(k))
    for k, v in part["E1_D2_labels"].items():
        arms["D2"][k] = dict(stage_name=v, source="label_only", stop=part["E1_D2_stop_reasons_seen"].get(k))
    for k, v in part["E2_D3_labels"].items():
        arms["D3"][k] = dict(stage_name=v.split(" ")[0], source="label_only", note=v)
    e3 = json.loads((args.docs / "study_E3.json").read_text())
    for t in e3["trials"]:
        arm = t["recipe"].split("_")[-1]
        f = args.e3_raw / "rollouts" / t["id"] / "trial_result.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        if r.get("status") not in ("scored", "integrity_stopped"):
            continue
        arms[arm][key(t["start"])] = dict(stage_name=r["stage_name"], source="E3:" + r["status"], stop=r.get("stop_reason"),
                                          dropped=r.get("dropped"), inside_bin=bool((r.get("object") or {}).get("final_inside_bin")),
                                          integrity_stopped=r.get("status") == "integrity_stopped")
    out = {"baseline": "Stage C v6:K4_ir (runtime v13)", "arms": {}}
    lines = ["| arm | starts | placed (arm vs baseline, same starts) | stage>=3 | mid-carry ceiling stops | ended in bin (E3 trials only) | better / worse / tie | sign test p |",
             "|---|---|---|---|---|---|---|---|"]
    for arm, d in arms.items():
        keys = sorted(k for k in d if k in base)
        st = {k: STAGE[d[k]["stage_name"]] for k in keys}
        bs = {k: STAGE[base[k]["stage_name"]] for k in keys}
        wins = sum(st[k] > bs[k] for k in keys); losses = sum(st[k] < bs[k] for k in keys)
        placed = sum(st[k] == 4 for k in keys); bplaced = sum(bs[k] == 4 for k in keys)
        box = sum(st[k] >= 3 for k in keys); bbox = sum(bs[k] >= 3 for k in keys)
        ceiling = sum(1 for k in keys if (d[k].get("stop") or "").startswith("boundary_projection_final_envelope"))
        bceiling = sum(1 for k in keys if (base[k].get("stop_reason") or "") == "boundary_projection_final_envelope")
        inbin = [k for k in keys if d[k].get("source", "").startswith("E3")]
        ended = sum(1 for k in inbin if d[k].get("inside_bin"))
        counts = Counter(d[k]["stage_name"] for k in keys)
        out["arms"][arm] = dict(n=len(keys), placed=placed, baseline_placed=bplaced, stage3plus=box, baseline_stage3plus=bbox,
                                ceiling_stops=ceiling, baseline_ceiling_stops=bceiling, wins=wins, losses=losses, ties=len(keys) - wins - losses,
                                sign_p=sign_test(wins, losses), counts=dict(counts), label_only=sum(1 for k in keys if d[k]["source"] == "label_only"),
                                integrity_stopped=sum(1 for k in keys if d[k].get("integrity_stopped")),
                                per_start={k: dict(arm=d[k], baseline=base[k]["stage_name"]) for k in keys})
        lines.append(f"| {arm} | {len(keys)} | {placed} vs {bplaced} | {box} vs {bbox} | {ceiling} vs {bceiling} | {ended} / {len(inbin)} | {wins} / {losses} / {len(keys)-wins-losses} | {sign_test(wins, losses)} |")
    md = "\n".join(lines)
    print(md)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "treatments.json").write_text(json.dumps(out, indent=2) + "\n")
        (args.output / "treatments.md").write_text(md + "\n")


if __name__ == "__main__":
    main()
