import json,glob,os,sys,math,collections
import numpy as np, zarr
ROOT=sys.argv[1]
A2,A3,D4,D1=0.24365,0.21325,0.11235,0.1519
def wrist_dist(q3): return math.sqrt(A2*A2+A3*A3+2*A2*A3*math.cos(q3)+D4*D4)
rows=[]
for d in sorted(glob.glob(ROOT+"/ep_*")):
    try:
        m=json.load(open(d+"/meta.json")); s=json.load(open(d+"/stop.json")) if os.path.exists(d+"/stop.json") else {}
        tags={t.split(":")[0]:t.split(":",1)[1] for t in m.get("tags",[]) if ":" in t}
        tcp=np.asarray(zarr.open_group(d+"/arm_tcp_pose.zarr",mode="r")["data"][:]); tts=np.asarray(zarr.open_group(d+"/arm_tcp_pose.zarr",mode="r")["ts"][:])
        q=np.asarray(zarr.open_group(d+"/arm_q.zarr",mode="r")["data"][:])
        g=np.asarray(zarr.open_group(d+"/gripper.zarr",mode="r")["data"][:]); gts=np.asarray(zarr.open_group(d+"/gripper.zarr",mode="r")["ts"][:])
        t0=tts[0]; t=tts-t0; dur=float(t[-1]) if len(t) else 0
        z=tcp[:,2]; r_h=np.hypot(tcp[:,0],tcp[:,1]); wd=np.array([wrist_dist(x) for x in q[:,2]])
        iapex=int(np.argmax(z)); 
        closed=g[:,0]>0.35 if len(g) else np.zeros(0,bool)
        ic=int(np.argmax(closed)) if closed.any() else None
        tc=float(gts[ic]-t0) if ic is not None else None
        zc=float(np.interp(gts[ic],tts,z)) if ic is not None else None
        # lift: after closure z rises > 5cm above closure z
        lifted=bool(ic is not None and (z[tts>=gts[ic]].max()-zc)>0.05) if ic is not None else False
        rows.append(dict(ep=os.path.basename(d),task=m.get("task"),ckpt=tags.get("ckpt"),seed=tags.get("seed"),notes=m.get("notes"),success=m.get("success"),
            stop=s.get("stopped_reason"),events=s.get("safety_events"),dur=round(dur,1),n_replans=s.get("n_replans"),
            apex_z=round(float(z[iapex]),3),apex_t=round(float(t[iapex]),1),apex_wd=round(float(wd[iapex]),3),apex_rh=round(float(r_h[iapex]),3),
            max_wd=round(float(wd.max()),3),max_rh=round(float(r_h.max()),3),max_elbow_min_deg=round(float(np.degrees(np.abs(q[:,2]).min())),1),
            stop_z=round(float(z[-1]),3),stop_wd=round(float(wd[-1]),3),stop_xy=[round(float(tcp[-1,0]),3),round(float(tcp[-1,1]),3)],
            close_t=None if tc is None else round(tc,1),close_z=None if zc is None else round(zc,3),lifted=lifted,
            z_at_max_wd=round(float(z[int(np.argmax(wd))]),3),t_at_max_wd=round(float(t[int(np.argmax(wd))]),1)))
    except Exception as e:
        rows.append(dict(ep=os.path.basename(d),error=repr(e)[:120]))
json.dump(rows,open(sys.argv[2],"w"),indent=1)
ok=[r for r in rows if "error" not in r]; print(len(rows),"episodes,",len(ok),"parsed,",len(rows)-len(ok),"errors")
by=collections.defaultdict(list)
for r in ok: by[(r["task"],r["ckpt"])].append(r)
for k,v in sorted(by.items(),key=lambda kv:-len(kv[1])):
    stops=collections.Counter((r["stop"],tuple(r["events"] or [])) for r in v)
    lift=[r for r in v if r["lifted"]]; succ=[r for r in v if r["success"]]
    print(f"\n== {k[0]} | {k[1]} | n={len(v)} | success={len(succ)} | lifted={len(lift)} | closed={sum(r['close_t'] is not None for r in v)}")
    print("   stops:",dict(stops))
    if v: print("   apex_z mean/max",round(np.mean([r['apex_z'] for r in v]),3),max(r['apex_z'] for r in v),"| max_wd mean/max",round(np.mean([r['max_wd'] for r in v]),3),max(r['max_wd'] for r in v),"| min elbow deg",min(r['max_elbow_min_deg'] for r in v))
    if lift: print("   LIFTED: apex_z",[r['apex_z'] for r in lift],"max_wd",[r['max_wd'] for r in lift],"stop",[ (r['stop'],r['events']) for r in lift],"notes",[r['notes'] for r in lift])
