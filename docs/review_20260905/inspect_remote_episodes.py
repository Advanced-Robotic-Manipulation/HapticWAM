"""Read-only compute3 episode audit. Run via SSH stdin; emits JSON only."""
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
import zarr

ROOT = Path('/home/physicalai/phantom-icra-2027/data/episodes/deploy')

def tag(meta, prefix):
    return next((s[len(prefix):] for s in meta.get('tags', []) if s.startswith(prefix)), None)

def stream(ep, name):
    path = ep / (name + '.zarr')
    if not path.exists():
        return None, None
    g = zarr.open(str(path), mode='r')
    return np.asarray(g['data']), np.asarray(g['ts'])

def num(x):
    return round(float(x), 6)

def longest(mask, ts):
    idx = np.flatnonzero(mask)
    if not len(idx):
        return 0.0
    groups = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    return num(max(ts[g[-1]] - ts[g[0]] for g in groups))

rows = []
for mp in sorted(ROOT.glob('*/ep_*/meta.json')):
    ep = mp.parent
    meta = json.loads(mp.read_text())
    stop_path = ep / 'stop.json'
    stop = json.loads(stop_path.read_text()) if stop_path.exists() else {}
    tp = ep / 'planner_trace.json'
    trace = json.loads(tp.read_text()) if tp.exists() else []
    lat = [t['latency_s'] for t in trace if t.get('latency_s') is not None]
    row = dict(date=ep.parent.name, episode=ep.name, remote_path=str(ep),
               meta=meta, stop=stop, meta_sha256=hashlib.sha256(mp.read_bytes()).hexdigest(),
               trace_sha256=hashlib.sha256(tp.read_bytes()).hexdigest() if tp.exists() else None,
               n_replans=len(trace), accepted_plans=sum(bool(t.get('accepted')) for t in trace),
               latency_p50_s=num(np.median(lat)) if lat else None,
               latency_p95_s=num(np.percentile(lat,95)) if lat else None,
               latency_max_s=num(max(lat)) if lat else None,
               trace_diag_variants=sorted(set(json.dumps({k:(t.get('diag') or {}).get(k) for k in ('nfe','guidance','k_seeds')},sort_keys=True) for t in trace)),
               stop_reason=stop.get('stopped_reason',tag(meta,'stop:')),
               safety_event_names=[e.get('kind',e.get('name',e.get('type','unknown'))) if isinstance(e,dict) else str(e) for e in stop.get('safety_events',[])],
               veto_actions=dict(collections.Counter((t.get('terminal_veto') or {}).get('action','unrecorded') for t in trace)))
    if ep.parent.name >= '20260901':
        try:
            pose, pts=stream(ep,'arm_tcp_pose')
            q, qts=stream(ep,'arm_q')
            qd, qdts=stream(ep,'arm_qd')
            gr, gts=stream(ep,'gripper')
            lw,lts=stream(ep,'tactile_left_wrench')
            rw,rts=stream(ep,'tactile_right_wrench')
            row['stream_counts']={n:len(t) if t is not None else 0 for n,t in [('pose',pts),('q',qts),('qd',qdts),('grip',gts),('left_wrench',lts),('right_wrench',rts)]}
            if pts is not None and len(pts):
                row.update(duration_s=num(pts[-1]-pts[0]),start_pose=pose[0].tolist(),end_pose=pose[-1].tolist(),
                           tcp_z_min_m=num(pose[:,2].min()),tcp_z_max_m=num(pose[:,2].max()),
                           tcp_y_min_m=num(pose[:,1].min()),tcp_y_max_m=num(pose[:,1].max()),
                           arm_hz=num(1/np.median(np.diff(pts))) if len(pts)>1 else None)
            if q is not None and len(q): row['elbow_abs_min_deg']=num(np.rad2deg(np.abs(q[:,2])).min())
            if qd is not None and len(qd): row['qd_abs_max_rad_s']=num(np.abs(qd).max())
            if gr is not None and len(gr):
                row.update(grip_pos_min=num(gr[:,0].min()),grip_pos_max=num(gr[:,0].max()),grip_pos_end=num(gr[-1,0]),
                           gripper_object_codes=sorted(set(gr[:,1].tolist())))
            if lts is not None and rts is not None and len(lts)>1 and len(rts)>1 and pts is not None and len(pts)>1:
                # One common clock, one bilateral event; diagnostic proxy only.
                left=np.abs(lw[:,2]);right=np.interp(lts,rts,np.abs(rw[:,2]))
                z=np.interp(lts,pts,pose[:,2]);y=np.interp(lts,pts,pose[:,1])
                row.update(tactile_left_hz=num(1/np.median(np.diff(lts))),tactile_right_hz=num(1/np.median(np.diff(rts))),
                           bilateral_gt3N_longest_s=longest((left>3)&(right>3),lts),
                           bilateral_gt3N_zgt320mm_longest_s=longest((left>3)&(right>3)&(z>.32),lts))
                row['sampled_trajectory']={}
                ix=np.unique(np.linspace(0,len(pts)-1,min(160,len(pts))).astype(int))
                row['sampled_trajectory']={'t_s':[num(x) for x in pts[ix]-pts[0]],'xyz_m':pose[ix,:3].round(6).tolist(),
                    'left_fz_N':np.interp(pts[ix],lts,left).round(3).tolist(),'right_fz_N':np.interp(pts[ix],rts,np.abs(rw[:,2])).round(3).tolist(),
                    'grip':np.interp(pts[ix],gts,gr[:,0]).round(4).tolist() if gts is not None and len(gts) else []}
        except Exception as exc:
            row['stream_error']=type(exc).__name__+': '+str(exc)
    rows.append(row)
print(json.dumps({'root':str(ROOT),'episodes':rows},allow_nan=False))
