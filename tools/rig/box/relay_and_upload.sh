#!/bin/bash
# Site-specific: host aliases and absolute paths below are our lab's -- adapt to your own setup.
# Rig box: pull the packed dataset from the training box over the VPN tunnel, then
# upload every file to the hub with per-file retries — each file to the repo
# that owns it after the 2026-09 restructure (see route() below).
# HF_TOKEN must be in the environment.
set -u
cd ~/phantom-icra-2027
echo "== rsync from the training box $(date)"
rsync -a --partial anywherevla@100.64.0.2:phantom-icra-2027/packed/ packed_relay/
echo "== rsync done $(date): $(ls packed_relay | wc -l) files, $(du -sh packed_relay | cut -f1)"

~/phantom-icra-2027/phantom/.venv/bin/python - <<'EOF'
import glob, os, time
from huggingface_hub import upload_file, list_repo_files

# The 2026-09 armteam restructure split the old dataset_v3_packed/ folder across
# several clean public repos, so the destination now depends on the file name.
# The new repos are FLAT (no dataset_v3_packed/ prefix) and every destination below
# is one of them. The source tarball is never published: provisioning clones the
# code from GitHub instead.
TELEOP = 'armteam/hapticwam-teleop-dataset'
ROLLOUTS = 'armteam/hapticwam-rollouts'
TEACHER = 'armteam/hapticwam-teacher'
BASELINES = 'armteam/hapticwam-baselines'
NEVER = {'phantom_repo_latest.tar.gz'}   # the source tree: never goes to the hub


def route(name):
    """file name -> (repo_id, repo_type, path_in_repo), or None if it must not be uploaded"""
    if name in NEVER:
        return None
    if name == 'text_embeddings.pt':
        return TEACHER, 'model', name
    if name == 'lerobot_phantom_pi05.tar.zst':
        # a derived export: it ships with the pi0.5 model it feeds
        return BASELINES, 'model', 'lerobot_export/' + name
    if name.startswith(('rollouts_', 'deploy_')) or name == 'manifests_r3.tar':
        return ROLLOUTS, 'dataset', name
    return TELEOP, 'dataset', name   # task tarballs, manifests*.tar, norm_stats.json, batch_*


files = sorted(glob.glob(os.path.expanduser('~/phantom-icra-2027/packed_relay/*')))
print('files to upload:', len(files), flush=True)
for f in files:
    name = os.path.basename(f)
    routed = route(name)
    if routed is None:
        print('skip (source tarball, never published):', name, flush=True)
        continue
    repo, rtype, dest = routed
    for attempt in range(60):
        try:
            done = set(list_repo_files(repo, repo_type=rtype))
            if dest in done:
                print('already-up', name, flush=True)
                break
            print('uploading', name, '->', repo, 'attempt', attempt, flush=True)
            upload_file(path_or_fileobj=f, path_in_repo=dest,
                        repo_id=repo, repo_type=rtype)
            print('done', name, flush=True)
            break
        except Exception as e:
            print('retry:', type(e).__name__, str(e)[:100], flush=True)
            time.sleep(min(120, 10 * (attempt + 1)))
    else:
        print('GAVE-UP', name, flush=True)
have = {}
missing = []
for f in files:
    name = os.path.basename(f)
    routed = route(name)
    if routed is None:
        continue
    repo, rtype, dest = routed
    if repo not in have:
        have[repo] = set(list_repo_files(repo, repo_type=rtype))
    if dest not in have[repo]:
        missing.append(name)
print('MISSING-AFTER-RUN:', missing, flush=True)
print('ALL-UPLOADED' if not missing else 'INCOMPLETE', flush=True)
EOF
