"""Read-only recomputation of archived PHANTOM terminal-evaluation artifacts.

Usage: python recompute_phantom_e13.py /path/to/copied/artifact-directory
Expected: e13_val{78,124}_{v4,v5_6}.json and manifests/{val_eval,val124}_all.jsonl.
Bootstrap units are episodes (four inference seeds remain together), with a
collection-session sensitivity analysis. These are descriptive, post-selection
validation intervals, not preregistered robot-success estimates.
"""
import collections
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

root = Path(sys.argv[1])

def read_json(path):
    return json.loads(path.read_text())

def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)

def mean(xs):
    xs = list(xs)
    return statistics.mean(xs) if xs else None

def quantile(xs, q):
    s = sorted(xs)
    p = (len(s) - 1) * q
    lo, hi = math.floor(p), math.ceil(p)
    return s[lo] * (hi - p) + s[hi] * (p - lo) if hi != lo else s[lo]

def bootstrap_clusters(values_by_cluster, seed=20260905, draws=10000):
    clusters = list(values_by_cluster.values())
    sums = [sum(v) for v in clusters]
    counts = [len(v) for v in clusters]
    rng = random.Random(seed)
    sampled = []
    for _ in range(draws):
        idxs = [rng.randrange(len(clusters)) for _ in clusters]
        sampled.append(sum(sums[i] for i in idxs) / sum(counts[i] for i in idxs))
    return {"clusters": len(clusters), "draws": draws, "seed": seed,
            "percentile95": [quantile(sampled, .025), quantile(sampled, .975)]}

old_manifest = [json.loads(l) for l in (root / 'manifests/val_eval_all.jsonl').read_text().splitlines() if l.strip()]
new_manifest = [json.loads(l) for l in (root / 'manifests/val124_all.jsonl').read_text().splitlines() if l.strip()]
session_by_ep = {r['episode']: r['session'] for r in new_manifest}
session_by_ep.update({r['episode']: r['session'] for r in old_manifest})
session_splits = collections.defaultdict(set)
for r in old_manifest:
    session_splits[r['session']].add(r['split'])
result = {"old_manifest": {"rows": len(old_manifest), "sessions": len(session_splits),
          "mixed_split_sessions": sorted(s for s, v in session_splits.items() if len(v) > 1),
          "sessions_by_split": dict(collections.Counter(next(iter(v)) for v in session_splits.values()))},
          "method": "v5_6 minus v4; negative endpoint delta favors v5; episode means over all four inference seeds; bootstrap retains clusters",
          "splits": {}, "artifacts": {}}
for split in ['val78', 'val124']:
    paths = [root / f'e13_{split}_{m}.json' for m in ['v4', 'v5_6']]
    docs = [read_json(p) for p in paths]
    maps = [{(r['episode'], r['t0'], r['seed']): r for r in d['rows']} for d in docs]
    assert maps[0].keys() == maps[1].keys(), 'mismatched terminal windows/seeds'
    for d, m in zip(docs, maps):
        assert len(d['rows']) == len(m), 'duplicate terminal rows'
    keys = sorted(maps[0])
    episode_deltas = collections.defaultdict(list)
    for key in keys:
        episode_deltas[key[0]].append(maps[1][key]['endpoint_err_mm'] - maps[0][key]['endpoint_err_mm'])
    assert all(len(v) == 4 for v in episode_deltas.values())
    ep_means = {ep: mean(v) for ep, v in episode_deltas.items()}
    session_deltas = collections.defaultdict(list)
    for ep, v in ep_means.items():
        session_deltas[session_by_ep[ep]].append(v)
    shared_close_keys = [key for key in keys if all(finite(m[key]['pred_close_height_mm']) and finite(m[key]['gt_close_height_mm']) for m in maps)]
    close = {}
    for name, doc, mp in zip(['v4', 'v5_6'], docs, maps):
        rs = doc['rows']
        own = [r['pred_close_height_mm'] - r['gt_close_height_mm'] for r in rs if finite(r['pred_close_height_mm']) and finite(r['gt_close_height_mm'])]
        gt_n = sum(finite(r['gt_close_height_mm']) for r in rs)
        predicted_n = sum(finite(r['pred_close_height_mm']) for r in rs)
        close[name] = {"finite_gt_close_rows": gt_n, "finite_predicted_close_rows": predicted_n,
                      "predicted_close_fraction_among_gt_close_rows": predicted_n / gt_n,
                      "no_predicted_close_rows_with_gt_close": gt_n - predicted_n,
                      "own_subset_close_error_mm": mean(own), "own_subset_n": len(own),
                      "shared_subset_close_error_mm": mean(mp[k]['pred_close_height_mm'] - mp[k]['gt_close_height_mm'] for k in shared_close_keys)}
    result['splits'][split] = {"rows": len(keys), "episodes": len(ep_means), "sessions": len(session_deltas),
        "endpoint_mean_mm": {name: mean(r['endpoint_err_mm'] for r in doc['rows']) for name, doc in zip(['v4', 'v5_6'], docs)},
        "episode_mean_delta_mm": mean(ep_means.values()), "episode_median_delta_mm": statistics.median(ep_means.values()),
        "episodes_v5_better": sum(v < 0 for v in ep_means.values()),
        "episode_cluster_bootstrap": bootstrap_clusters({ep: [v] for ep, v in ep_means.items()}),
        "session_cluster_bootstrap": bootstrap_clusters(session_deltas),
        "shared_finite_close_rows": len(shared_close_keys), "close": close}
    for p in paths:
        result['artifacts'][p.name] = {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}
result['e9'] = {}
for mode in ['none', 'tactile', 'wrist', 'prev_cpk', 'contact_zero', 'contact_gt']:
    p = root / f'e9_{mode}.json'
    if p.exists():
        d = read_json(p)
        result['e9'][mode] = {k: d['summary'].get(k) for k in ['n_episodes', 'n', 'endpoint_err_mm', 'commit_ratio', 'close_step_err', 'null_semantics']}
        result['artifacts'][p.name] = {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}
result['fta'] = {}
for p in sorted((root / 'ftA_score').glob('e13_ftA_*.json')):
    d = read_json(p); s = d['summary']
    result['fta'][p.stem] = {k: s[k] for k in ['n_episodes', 'n', 'endpoint_err_mm', 'median_endpoint_err_mm', 'z_end_err_mm', 'commit_ratio']}
    result['fta'][p.stem]['waffles_endpoint_err_mm'] = s['per_task']['waffles']['endpoint_err_mm']
    result['artifacts']['ftA_score/' + p.name] = {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}
print(json.dumps(result, indent=2, allow_nan=False))
