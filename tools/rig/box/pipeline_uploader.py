"""Watcher: upload each fully-relayed tarball while rsync keeps pulling.

A file is 'complete' when it has its final name (rsync --partial writes
dot-temp names) and its size matches the source manifest sizes. The main
relay script's later upload phase sees these as already-up and skips.
"""
import glob, os, time
from huggingface_hub import upload_file, list_repo_files

# The 2026-09 armteam restructure split the old dataset_v3_packed/ folder across
# several clean public repos, so the destination depends on the file name. The new
# repos are FLAT, with no dataset_v3_packed/ prefix, and every destination below is
# one of them. The source tarball is never published anywhere: the provisioning
# scripts clone the code from GitHub instead.
TELEOP = "armteam/hapticwam-teleop-dataset"
ROLLOUTS = "armteam/hapticwam-rollouts"
TEACHER = "armteam/hapticwam-teacher"
BASELINES = "armteam/hapticwam-baselines"
REPO_TYPES = {TELEOP: "dataset", ROLLOUTS: "dataset",
              TEACHER: "model", BASELINES: "model"}
NEVER = {"phantom_repo_latest.tar.gz"}   # the source tree: never goes to the hub


def route(name):
    """file name -> (repo_id, path_in_repo), or None if it must not be uploaded"""
    if name in NEVER:
        return None
    if name == "text_embeddings.pt":
        return TEACHER, name
    if name == "lerobot_phantom_pi05.tar.zst":
        # a derived export: it ships with the pi0.5 model it feeds
        return BASELINES, "lerobot_export/" + name
    if name.startswith(("rollouts_", "deploy_")) or name == "manifests_r3.tar":
        return ROLLOUTS, name
    return TELEOP, name   # task tarballs, manifests*.tar, norm_stats.json, batch_*


EXPECT = {}  # name -> size, filled from sizes.txt written by the relay kick
with open(os.path.expanduser("~/phantom-icra-2027/relay_sizes.txt")) as fh:
    for line in fh:
        sz, name = line.split(None, 1)
        EXPECT[name.strip()] = int(sz)

uploaded = set()
idle = 0
while len(uploaded) < len(EXPECT) and idle < 120:
    progress = False
    try:
        have = {r: set(list_repo_files(r, repo_type=t))
                for r, t in REPO_TYPES.items()}
    except Exception:
        time.sleep(30)
        continue
    for f in sorted(glob.glob(os.path.expanduser("~/phantom-icra-2027/packed_relay/*"))):
        name = os.path.basename(f)
        if name.startswith(".") or name in uploaded or name not in EXPECT:
            continue
        if os.path.getsize(f) != EXPECT[name]:
            continue                        # still transferring
        routed = route(name)
        if routed is None:
            uploaded.add(name)   # counts as handled so the watcher can finish
            print("skip (source tarball, never published):", name, flush=True)
            continue
        repo, dest = routed
        if dest in have[repo]:
            uploaded.add(name)
            continue
        print("uploading", name, "->", repo, flush=True)
        try:
            upload_file(path_or_fileobj=f, path_in_repo=dest,
                        repo_id=repo, repo_type=REPO_TYPES[repo])
            uploaded.add(name)
            print("done", name, flush=True)
            progress = True
        except Exception as e:
            print("retry-later:", name, type(e).__name__, str(e)[:80], flush=True)
            time.sleep(20)
    idle = 0 if progress else idle + 1
    time.sleep(30)
print("WATCHER-COMPLETE", sorted(uploaded), flush=True)
