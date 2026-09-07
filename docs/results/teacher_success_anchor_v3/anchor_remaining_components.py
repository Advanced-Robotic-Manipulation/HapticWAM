"""Execute only the remaining four frozen single-component diagnostics."""
from pathlib import Path
import os
import subprocess
import json
import hashlib
import datetime

b = Path('/home/physicalai/phantom-icra-2027/sim/waffles')
r = b / 'runs/teacher_success_anchor_v3'
live = Path('/home/physicalai/phantom-icra-2027/phantom')
variants = [
    ('gel_v2_only', 'source_teacher_anchor_gel_v3'),
    ('wrist_distal_only', 'source_teacher_anchor_wrist_v3'),
    ('live_veto_only', 'source_teacher_pick_place_v1'),
    ('delivery_feedback_only', 'source_teacher_anchor_feedback_v3'),
]
env = os.environ.copy()
env.update(PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
for variant, source in variants:
    s = b / source
    campaign = r / 'component_campaigns' / (variant + '.json')
    spec = json.loads(campaign.read_text())
    assert spec['anchor_component']['source'] == str(s)
    assert spec['sampling_seeds'] == [904301, 904302]
    assert spec['planned_counts']['total'] == 2
    cmd = [str(live / '.venv/bin/python'), str(s / 'tools/sim/run_policy_campaign.py'),
           '--campaign', str(campaign), '--source', str(s), '--live-repo', str(live),
           '--evidence', str(b / 'evidence'), '--hardware-config',
           str(b / 'runs/teacher_pick_place_v1/runtime/hardware_campaign.yaml'),
           '--output', str(r / 'components' / variant), '--robot-usd',
           str(b / 'runs/validation_v2/pick_place/robot_asset/waffles/waffles.usda'),
           '--port', '7799']
    if (r / 'components' / variant).exists():
        raise FileExistsError('Preserve attempted diagnostics; do not retry automatically')
    subprocess.run(cmd, check=True, env=env)
    with (r / ('component_' + variant + '.log')).open('w') as log:
        p = subprocess.Popen(cmd + ['--execute'], stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True, env=env, cwd=s)
        (r / ('component_' + variant + '_launcher.json')).write_text(json.dumps({
            'pid': p.pid, 'command': cmd + ['--execute'],
            'campaign_sha256': hashlib.sha256(campaign.read_bytes()).hexdigest(),
            'launched_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }, indent=2) + '\n')
        print('Started', variant, p.pid, flush=True)
        code = p.wait()
    if code:
        raise RuntimeError(f'{variant} failed with code {code}; preserve logs')
    summary = json.loads((r / 'components' / variant / 'analysis/summary.json').read_text())
    assert summary['status'] == 'complete'
    print('Completed', variant, flush=True)
