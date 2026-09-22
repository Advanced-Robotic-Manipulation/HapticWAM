# Site-specific: host aliases and absolute paths below are our lab's -- adapt to your own setup.
import json,glob,numpy as np,zarr,math
def load(d):
    tcp=np.asarray(zarr.open_group(d+"/arm_tcp_pose.zarr",mode="r")["data"][:]); ts=np.asarray(zarr.open_group(d+"/arm_tcp_pose.zarr",mode="r")["ts"][:])
    q=np.asarray(zarr.open_group(d+"/arm_q.zarr",mode="r")["data"][:]); n=min(len(q),len(tcp)); ok=np.abs(q[:n]).sum(1)>1e-6
    ft=None
    try:
        f=zarr.open_group(d+"/arm_ft.zarr",mode="r"); ft=np.asarray(f["data"][:]); fts=np.asarray(f["ts"][:])
    except Exception: fts=None
    return tcp[:n][ok],ts[:n][ok],ft,fts
def cross(label,d):
    tcp,ts,ft,fts=load(d); z=tcp[:,2]; y=tcp[:,1]
    j=int(np.argmax(z)); post=slice(j,len(tcp))
    out=[]
    for yth in (0.0,0.03,0.06):
        idx=np.where(y[post]>yth)[0]
        out.append(f"y>{yth:+.2f}: z={z[j+idx[0]]:.3f} x={tcp[j+idx[0],0]:.3f}" if len(idx) else f"y>{yth:+.2f}: never")
    iy=int(np.argmax(y)); 
    s=f"{label} {d[-22:]} apex z {z[j]:.3f} | "+" | ".join(out)+f" | max y {y.max():.3f} at z {z[iy]:.3f} x {tcp[iy,0]:.3f} | end ({tcp[-1,0]:.2f},{tcp[-1,1]:.2f},{tcp[-1,2]:.2f})"
    if ft is not None and len(ft):
        k=len(ft)-1; fz=ft[max(0,k-20):k+1,:3].mean(0); s+=f" | end force xyz {np.round(fz,1).tolist()} |F| {np.linalg.norm(fz):.0f}N"
    print(s)
demos=[d for d in sorted(glob.glob("/home/physicalai/phantom-icra-2027/data/full/tasks/waffles/ep_*")) if any(t.startswith("batch") for t in json.load(open(d+"/meta.json")).get("tags",[]))]
for d in demos: cross("DEMO",d)
rows=json.load(open("/tmp/apex_0911.json"))
for r in rows:
    if r.get("task")=="waffles" and r.get("lifted") and r.get("ckpt")=="teacher_020000.pt":
        cross("POL "+str([e["kind"] for e in (r["events"] or [])])+" "+str(r["notes"]),"/home/physicalai/phantom-icra-2027/data/episodes/deploy/20260911/"+r["ep"])
