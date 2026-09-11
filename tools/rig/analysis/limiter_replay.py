import json,glob,numpy as np,zarr,math,sys,collections
sys.path.insert(0,"/home/physicalai/phantom-icra-2027/phantom")
from phantom.sim.kinematics import inverse_kinematics
from phantom.drivers import servo_limiter as SL
A2,A3,D4=0.24365,0.21325,0.11235
wd=lambda q3: math.sqrt(A2*A2+A3*A3+2*A2*A3*math.cos(q3)+D4*D4)
limits=SL.ServoLimits(elbow_min_rad=0.40,joint_speed_max_rad_s=1.0,branch_tolerance_rad=0.35,bisection_iterations=3,shoulder_height_m=0.1519)
def solve_ik(pose,seed):
    r=inverse_kinematics(pose,seed)
    return list(r.q) if r.success else []
rows=json.load(open("/tmp/apex_0911.json"))
eps=["/home/physicalai/phantom-icra-2027/data/episodes/deploy/20260911/"+r["ep"] for r in rows if r.get("task")=="waffles" and r.get("lifted") and r.get("ckpt")=="teacher_020000.pt"]
for zcap in (None,0.35):
    print(f"\n##### bounded_v1 replay, zcap={zcap}")
    for d in eps:
        tcp=np.asarray(zarr.open_group(d+"/arm_tcp_pose.zarr",mode="r")["data"][:]); ts=np.asarray(zarr.open_group(d+"/arm_tcp_pose.zarr",mode="r")["ts"][:]); q=np.asarray(zarr.open_group(d+"/arm_q.zarr",mode="r")["data"][:])
        n=min(len(q),len(tcp)); tcp,q,ts=tcp[:n],q[:n],ts[:n]
        # 125Hz recorded; simulate at 8ms using every sample
        prev=tcp[0].copy(); qref=list(q[0]); modes=collections.Counter(); wds=[]; lags=[]; holds_run=0; max_hold_run=0; calls=0
        for k in range(1,n):
            tgt=tcp[k].copy()
            if zcap is not None and tgt[2]>zcap: tgt[2]=zcap
            dt=max(1e-3,float(ts[k]-ts[k-1]))
            st=SL.select_servo_step(tgt,prev,qref,dt,solve_ik,limits); calls+=st.ik_calls
            modes[st.mode if st.accepted else "hold:"+st.reason]+=1
            if st.accepted:
                prev=np.asarray(st.pose); qref=list(st.q); holds_run=0
            else:
                holds_run+=1; max_hold_run=max(max_hold_run,holds_run)
            wds.append(wd(qref[2])); lags.append(float(np.linalg.norm(prev[:3]-tgt[:3])))
        wds=np.array(wds); lags=np.array(lags)
        print(f"{d[-22:]} | modes {dict(modes)} | max wd {wds.max():.3f} (rec {max(wd(x) for x in q[:,2]):.3f}) | lag mean {lags.mean()*1000:.0f}mm max {lags.max()*1000:.0f}mm | longest hold run {max_hold_run} ticks | ik calls/tick {calls/(n-1):.2f}")
