import zarr, numpy as np, cv2, sys, os, glob
root="/home/physicalai/phantom-icra-2027/data/episodes/deploy/20260828"
names=["ep_teacher_waffles_1787922904_000","ep_teacher_waffles_1787929940_000","ep_teacher_waffles_1787944578_002","ep_teacher_whiteboard_1787939258_000","ep_teacher_Carton_1787944930_002","ep_teacher_waffles_1787945153_000"]
out="/tmp/frames"; os.makedirs(out, exist_ok=True)
tiles=[]
for n in names:
    g=zarr.open(f"{root}/{n}/camera_scene_color.zarr",mode="r"); d=g["data"]; T=d.shape[0]
    fr=[]
    for i in (0, T//3, 2*T//3, T-1):
        x=np.asarray(d[i])
        x=cv2.imdecode(np.frombuffer(x.tobytes(), np.uint8), cv2.IMREAD_COLOR)
        x=cv2.resize(x,(320,240)); cv2.putText(x,f"{n[11:30]} f{i}/{T}",(4,14),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,255),1); fr.append(x)
    tiles.append(np.hstack(fr))
img=np.vstack(tiles); cv2.imwrite(f"{out}/tiles.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70]); print(img.shape, d.dtype, d.shape)
