#!/bin/bash
# compute3: pull packed dataset from compute over the headscale tunnel, then
# upload every file to the private hub with per-file retries.
# HF_TOKEN must be in the environment.
set -u
cd ~/phantom-icra-2027
echo "== rsync from compute $(date)"
rsync -a --partial anywherevla@100.64.0.2:phantom-icra-2027/packed/ packed_relay/
echo "== rsync done $(date): $(ls packed_relay | wc -l) files, $(du -sh packed_relay | cut -f1)"

~/phantom-icra-2027/phantom/.venv/bin/python - <<'EOF'
import glob, os, time
from huggingface_hub import upload_file, list_repo_files
files = sorted(glob.glob(os.path.expanduser('~/phantom-icra-2027/packed_relay/*')))
print('files to upload:', len(files), flush=True)
for f in files:
    name = os.path.basename(f)
    dest = 'dataset_v3_packed/' + name
    for attempt in range(60):
        try:
            done = set(list_repo_files('armteam/phantom-checkpoints', repo_type='model'))
            if dest in done:
                print('already-up', name, flush=True)
                break
            print('uploading', name, 'attempt', attempt, flush=True)
            upload_file(path_or_fileobj=f, path_in_repo=dest,
                        repo_id='armteam/phantom-checkpoints', repo_type='model')
            print('done', name, flush=True)
            break
        except Exception as e:
            print('retry:', type(e).__name__, str(e)[:100], flush=True)
            time.sleep(min(120, 10 * (attempt + 1)))
    else:
        print('GAVE-UP', name, flush=True)
have = set(list_repo_files('armteam/phantom-checkpoints', repo_type='model'))
missing = [os.path.basename(f) for f in files
           if 'dataset_v3_packed/' + os.path.basename(f) not in have]
print('MISSING-AFTER-RUN:', missing, flush=True)
print('ALL-UPLOADED' if not missing else 'INCOMPLETE', flush=True)
EOF
