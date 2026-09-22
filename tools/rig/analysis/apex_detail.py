# Site-specific: host aliases and absolute paths below are our lab's -- adapt to your own setup.
import json,glob,os,sys,math
import numpy as np, zarr
A2,A3,D4=0.24365,0.21325,0.11235
wd=lambda q3: math.sqrt(A2*A2+A3*A3+2*A2*A3*math.cos(q3)+D4*D4)
rows=json.load(open("/tmp/apex_0911.json"))
ROOT="/home/physicalai/phantom-icra-2027/data/episodes/deploy/20260911"
sel=[r for r in rows if r.get("task")=="waffles" and r.get("lifted")]
print("lifted waffles episodes:",len(sel))
for r in sel:
    d=ROOT+"/"+r["ep"]
    tcp=np.asarray(zarr.open_group(d+"/arm_tcp_pose.zarr",mode="r")["data"][:]); ts=np.asarray(zarr.open_group(d+"/arm_tcp_pose.zarr",mode="r")["ts"][:]); t=ts-ts[0]
    q=np.asarray(zarr.open_group(d+"/arm_q.zarr",mode="r")["data"][:]); w=np.array([wd(x) for x in q[:,2]])
    z=tcp[:,2]; rh=np.hypot(tcp[:,0],tcp[:,1])
    i=int(np.argmax(w)); j=int(np.argmax(z))
    print(f"\n{r['ep'][-22:]} {r['ckpt']} seed{r['seed']} | {r['notes']} | stop={r['stop']} {[e['kind'] for e in (r['events'] or [])]} dur={r['dur']}")
    print(f"  close t={r['close_t']} z={r['close_z']} | apex z={z[j]:.3f} @t={t[j]:.1f} xy=({tcp[j,0]:.3f},{tcp[j,1]:.3f}) rh={rh[j]:.3f} wd={w[j]:.3f} | max wd={w[i]:.3f} @t={t[i]:.1f} z={z[i]:.3f} xy=({tcp[i,0]:.3f},{tcp[i,1]:.3f}) rh={rh[i]:.3f} | stop z={z[-1]:.3f} xy=({tcp[-1,0]:.3f},{tcp[-1,1]:.3f}) wd={w[-1]:.3f}")
    # samples after closure every 1 s
    if r["close_t"] is not None:
        ks=[int(np.searchsorted(t,tt)) for tt in np.arange(r["close_t"],t[-1],1.0)]
        print("  post-close (t,z,rh,wd):"," ".join(f"({t[k]:.0f},{z[k]:.2f},{rh[k]:.2f},{w[k]:.3f})" for k in ks if k<len(t)))
