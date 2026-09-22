# Site-specific: host aliases and absolute paths below are our lab's -- adapt to your own setup.
import math, yaml
import rtde_receive
cfg = yaml.safe_load(open("/home/physicalai/phantom-icra-2027/phantom/configs/start_poses.yaml"))
task = cfg["tasks"]["waffles"] if "tasks" in cfg else cfg["waffles"]
qm, qs = task["q_mean"], task["q_std"]
r = rtde_receive.RTDEReceiveInterface("192.168.88.56")
q = r.getActualQ()
tcp = r.getActualTCPPose()
print("current q (deg):", [round(math.degrees(v),1) for v in q])
print("tcp:", [round(v,3) for v in tcp])
names = ["base","shoulder","elbow","wrist1","wrist2","wrist3"]
for i in range(6):
    std = max(qs[i], math.radians(5))
    sig = (q[i]-qm[i])/std
    flag = " <<< OUT" if abs(sig) > 2.5 else ""
    turn = " !!! FULL-TURN?" if abs(q[i]-qm[i]) > math.radians(300) else ""
    print(f"{names[i]:9s} live {math.degrees(q[i]):8.1f}  demo {math.degrees(qm[i]):8.1f}  sigma {sig:7.1f}{flag}{turn}")
