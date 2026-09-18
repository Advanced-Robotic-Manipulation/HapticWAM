"""Watcher: upload each fully-relayed tarball while rsync keeps pulling.

A file is 'complete' when it has its final name (rsync --partial writes
dot-temp names) and its size matches the source manifest sizes. The main
relay script's later upload phase sees these as already-up and skips.
"""
import glob, os, time
from huggingface_hub import upload_file, list_repo_files

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
        have = set(list_repo_files("armteam/phantom-checkpoints", repo_type="model"))
    except Exception:
        time.sleep(30)
        continue
    for f in sorted(glob.glob(os.path.expanduser("~/phantom-icra-2027/packed_relay/*"))):
        name = os.path.basename(f)
        if name.startswith(".") or name in uploaded or name not in EXPECT:
            continue
        if os.path.getsize(f) != EXPECT[name]:
            continue                        # still transferring
        dest = "dataset_v3_packed/" + name
        if dest in have:
            uploaded.add(name)
            continue
        print("uploading", name, flush=True)
        try:
            upload_file(path_or_fileobj=f, path_in_repo=dest,
                        repo_id="armteam/phantom-checkpoints", repo_type="model")
            uploaded.add(name)
            print("done", name, flush=True)
            progress = True
        except Exception as e:
            print("retry-later:", name, type(e).__name__, str(e)[:80], flush=True)
            time.sleep(20)
    idle = 0 if progress else idle + 1
    time.sleep(30)
print("WATCHER-COMPLETE", sorted(uploaded), flush=True)
