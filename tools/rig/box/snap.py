# Site-specific: host aliases and absolute paths below are our lab's -- adapt to your own setup.
import sys, glob, io, os, numpy as np
import imageio.v2 as iio
task = sys.argv[1]
try:
    import pyrealsense2 as rs
    p = rs.pipeline(); c = rs.config()
    c.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 15)
    p.start(c)
    # warm-up: keep draining frames for a few seconds so auto-exposure/white balance settle
    import time
    warm = float(os.environ.get("SNAP_WARMUP_S", "2.5"))
    t0 = time.time()
    n = 0
    while time.time() - t0 < warm or n < 8:
        f = p.wait_for_frames(3000); n += 1
    cur = np.asanyarray(f.get_color_frame().get_data()).copy()
    p.stop()
    src = "LIVE camera"
except Exception:
    import zarr
    eps = sorted(glob.glob("/home/physicalai/phantom-icra-2027/data/episodes/deploy/*/ep_*"), key=os.path.getmtime)
    z = zarr.open(eps[-1] + "/camera_scene_color.zarr")["data"]
    cur = iio.imread(io.BytesIO(bytes(z[len(z) - 1])))
    src = "latest RECORDED frame (camera busy)"
ref = iio.imread(f"/home/physicalai/phantom-icra-2027/refs/ref_{task}.png")
from PIL import Image, ImageDraw
def lab(a, t):
    im = Image.fromarray(a); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 26], fill=(0, 0, 0))
    d.text((8, 5), t, fill=(255, 255, 0))
    return np.asarray(im)
h = min(cur.shape[0], ref.shape[0])
grid = np.hstack([lab(np.asarray(ref)[:h], f"TRAINING reference: {task}"),
                  lab(np.asarray(cur)[:h], f"NOW ({src})")])
iio.imwrite("/tmp/snap.png", grid)
print("snapshot source:", src)
