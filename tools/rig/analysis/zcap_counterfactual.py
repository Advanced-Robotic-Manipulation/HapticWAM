import json,glob,numpy as np,zarr,math,sys
sys.path.insert(0,"/home/physicalai/phantom-icra-2027/phantom")
from phantom.sim.kinematics import forward_pose, inverse_kinematics
A2,A3,D4=0.24365,0.21325,0.11235
wd=lambda q3: math.sqrt(A2*A2+A3*A3+2*A2*A3*math.cos(q3)+D4*D4)
rows=json.load(open("/tmp/apex_0911.json"))
eps=["/home/physicalai/phantom-icra-2027/data/episodes/deploy/20260911/"+r["ep"] for r in rows if r.get("task")=="waffles" and r.get("lifted") and r.get("ckpt")=="teacher_020000.pt"]
import inspect; print("IK sig:",inspect.signature(inverse_kinematics))
for d in eps:
    tcp=np.asarray(zarr.open_group(d+"/arm_tcp_pose.zarr",mode="r")["data"][:]); q=np.asarray(zarr.open_group(d+"/arm_q.zarr",mode="r")["data"][:]); n=min(len(q),len(tcp)); tcp,q=tcp[:n],q[:n]
    w=np.array([wd(x) for x in q[:,2]])
    out=[f"{d[-22:]} recorded max wd {w.max():.3f}"]
    for zcap in (0.36,0.35,0.34,0.33):
        wmax=0; fails=0; worst=None; sel=range(0,n,25)
        for k in sel:
            pose=tcp[k].copy()
            if pose[2]<=zcap: 
                wmax=max(wmax,w[k]); continue
            pose[2]=zcap
            try:
                r=inverse_kinematics(pose,q[k])
                qq=r.q if hasattr(r,"q") else r
                if qq is None or (hasattr(r,"ok") and not r.ok): fails+=1; continue
                v=wd(qq[2]); 
                if v>wmax: wmax=v; worst=(round(float(pose[0]),3),round(float(pose[1]),3))
            except Exception as e: fails+=1
        out.append(f"zcap {zcap}: max wd {wmax:.3f} ik_fail {fails}/{len(sel)} worst_xy {worst}")
    print(" | ".join(out))
