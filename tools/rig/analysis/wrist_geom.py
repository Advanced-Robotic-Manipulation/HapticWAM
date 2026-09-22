# Site-specific: host aliases and absolute paths below are our lab's -- adapt to your own setup.
import json,glob,numpy as np,zarr,math,sys
sys.path.insert(0,"/home/physicalai/phantom-icra-2027/phantom")
from phantom.sim.kinematics import dh_frames, forward_pose
A2,A3,D4=0.24365,0.21325,0.11235
wd=lambda q3: math.sqrt(A2*A2+A3*A3+2*A2*A3*math.cos(q3)+D4*D4)
def load(d):
    tcp=np.asarray(zarr.open_group(d+"/arm_tcp_pose.zarr",mode="r")["data"][:]); ts=np.asarray(zarr.open_group(d+"/arm_tcp_pose.zarr",mode="r")["ts"][:])
    q=np.asarray(zarr.open_group(d+"/arm_q.zarr",mode="r")["data"][:]); n=min(len(q),len(tcp)); ok=np.abs(q[:n]).sum(1)>1e-6
    g=zarr.open_group(d+"/gripper.zarr",mode="r"); gd=np.asarray(g["data"][:]); gts=np.asarray(g["ts"][:]) if "ts" in g else None
    return tcp[:n][ok],ts[:n][ok],q[:n][ok],gd,gts
def geom(q):
    F=dh_frames(q); sh=F[1][:3,3]; el=F[2][:3,3]; wr=F[3][:3,3]; w2=F[5][:3,3]
    return dict(shoulder=sh,elbow=el,wrist1=wr,wrist2=w2)
def summarize(label,eps):
    rows=[]
    for d in eps:
        try:
            tcp,ts,q,gd,gts=load(d)
            fk=np.array([forward_pose(qq)[:3] for qq in q[::200]]); err=np.linalg.norm(fk-tcp[::200][:len(fk),:3],axis=1)
            w=np.array([wd(x) for x in q[:,2]]); i=int(np.argmax(w)); j=int(np.argmax(tcp[:,2]))
            gi=geom(q[i]); gj=geom(q[j])
            rows.append(dict(ep=d[-22:],fk_err_mm=float(np.median(err))*1000,ymin=float(tcp[:,1].min()),ymax=float(tcp[:,1].max()),
                maxwd=float(w[i]),maxwd_tcp=tcp[i,:3].round(3).tolist(),maxwd_elbow_rh=float(np.hypot(*gi["elbow"][:2])),maxwd_elbow_z=float(gi["elbow"][2]),maxwd_wrist_rh=float(np.hypot(*gi["wrist1"][:2])),maxwd_wrist_z=float(gi["wrist1"][2]),maxwd_w2_rh=float(np.hypot(*gi["wrist2"][:2])),maxwd_w2_z=float(gi["wrist2"][2]),
                apex_tcp=tcp[j,:3].round(3).tolist(),apex_wd=float(w[j]),apex_wrist_rh=float(np.hypot(*gj["wrist1"][:2])),apex_wrist_z=float(gj["wrist1"][2]),end=tcp[-1,:3].round(3).tolist()))
        except Exception as e: rows.append(dict(ep=d[-22:],error=repr(e)[:80]))
    print("==",label)
    for r in rows:
        if "error" in r: print("  ",r); continue
        print(f"  {r['ep']} fkerr {r['fk_err_mm']:.0f}mm | y[{r['ymin']:.2f},{r['ymax']:.2f}] | MAXWD {r['maxwd']:.3f} tcp {r['maxwd_tcp']} elbow(rh {r['maxwd_elbow_rh']:.2f},z {r['maxwd_elbow_z']:.2f}) wrist1(rh {r['maxwd_wrist_rh']:.2f},z {r['maxwd_wrist_z']:.2f}) wrist2(rh {r['maxwd_w2_rh']:.2f},z {r['maxwd_w2_z']:.2f}) | APEX tcp {r['apex_tcp']} wd {r['apex_wd']:.3f} wrist1(rh {r['apex_wrist_rh']:.2f},z {r['apex_wrist_z']:.2f}) | end {r['end']}")
demos=[d for d in sorted(glob.glob("/home/physicalai/phantom-icra-2027/data/full/tasks/waffles/ep_*")) if any(t.startswith("batch") for t in json.load(open(d+"/meta.json")).get("tags",[]))]
summarize("DEMO batch_20260822",demos)
rows=json.load(open("/tmp/apex_0911.json"))
summarize("POLICY v6 lifted",["/home/physicalai/phantom-icra-2027/data/episodes/deploy/20260911/"+r["ep"] for r in rows if r.get("task")=="waffles" and r.get("lifted") and r.get("ckpt")=="teacher_020000.pt"])
