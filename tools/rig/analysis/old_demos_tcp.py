import json,glob,numpy as np,zarr,collections
rows=[]
for d in sorted(glob.glob("/home/physicalai/phantom-icra-2027/data/full/tasks/waffles/ep_*")):
    m=json.load(open(d+"/meta.json")); batch=[t for t in m.get("tags",[]) if t.startswith("batch")]
    tcp=np.asarray(zarr.open_group(d+"/arm_tcp_pose.zarr",mode="r")["data"][:]); tcp=tcp[np.abs(tcp).sum(1)>1e-6]
    if len(tcp)<200: continue
    z=tcp[:,2]; y=tcp[:,1]; j=int(np.argmax(z)); iy=int(np.argmax(y))
    idx6=np.where(y[j:]>0.06)[0]; idx3=np.where(y[j:]>0.03)[0]; idx0=np.where(y[j:]>0.0)[0]
    rows.append(dict(batch=batch[0] if batch else "old",sess=d.split("_")[-2][:6],apex=float(z[j]),x_apex=float(tcp[j,0]),y_apex=float(tcp[j,1]),maxy=float(y.max()),z_maxy=float(z[iy]),x_maxy=float(tcp[iy,0]),
        z_y0=float(z[j+idx0[0]]) if len(idx0) else np.nan,z_y03=float(z[j+idx3[0]]) if len(idx3) else np.nan,z_y06=float(z[j+idx6[0]]) if len(idx6) else np.nan,zmin_after_apex=float(z[j:].min()),ymin=float(y.min()),xmin=float(tcp[:,0].min())))
print(len(rows),"demos")
def rep(label,v):
    A=lambda k: np.array([r[k] for r in v])
    print(f"== {label} n={len(v)} | apex z mean {A('apex').mean():.3f} p10/p90 {np.percentile(A('apex'),10):.3f}/{np.percentile(A('apex'),90):.3f} at (x {A('x_apex').mean():.3f}, y {A('y_apex').mean():.3f}) | max y mean {A('maxy').mean():.3f} p90 {np.percentile(A('maxy'),90):.3f} (z {A('z_maxy').mean():.3f}, x {A('x_maxy').mean():.3f}) | z@y>0: {np.nanmean(A('z_y0')):.3f} z@y>0.03: {np.nanmean(A('z_y03')):.3f} z@y>0.06: {np.nanmean(A('z_y06')):.3f} (n {np.isfinite(A('z_y06')).sum()}) | x min {A('xmin').mean():.3f}")
by=collections.defaultdict(list)
for r in rows: by[r["sess"]].append(r)
for s,v in sorted(by.items()): rep("session "+s+" "+v[0]["batch"],v)
rep("ALL",rows)
