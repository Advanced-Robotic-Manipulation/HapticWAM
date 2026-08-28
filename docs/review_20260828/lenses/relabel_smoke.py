import logging, sys, time
logging.basicConfig(level=logging.WARNING)
from pathlib import Path
from phantom.config.hardware import load_hardware
from phantom.config.paths import load_paths
from phantom.train import common as C
from phantom.dagger.relabel import relabel_root
from phantom.dagger.manifest import write_manifest, read_manifest
hw = load_hardware()
paths = load_paths()
root = C.ensure_synthetic_dataset(paths.data_root / "synthetic", hw)
print("synthetic root", root, "eps", len(list(root.rglob('meta.json'))))
ck = Path("runs/teacher/v4smoke/teacher_000002.pt")
t=time.time()
n = relabel_root(root, ck, hw, paths, device="cpu", tiny=True)
print("relabeled", n, "eps in %.1fs" % (time.time()-t))
import torch
for ep in sorted(root.rglob("relabels/teacher.pt")):
    p = torch.load(ep, weights_only=False)
    r = p["records"]
    print(ep.parent.parent.name, "records", len(r), "t0s", [round(x["t0"],2) for x in r][:6], "actions", tuple(r[0]["actions"].shape))
    break
m = write_manifest(Path("/private/tmp/claude-501/-Users-sannikov/a5af36c2-8450-43d1-9a34-fa0cdb820706/scratchpad/review/round_1.json"), demos=root, rollouts=root, round_k=1, rollout_weight=0.5)
print("manifest entries", len(read_manifest(m)), read_manifest(m)[0])
