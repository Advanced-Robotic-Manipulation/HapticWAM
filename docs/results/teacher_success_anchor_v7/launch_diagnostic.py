"""Freeze and launch exactly four simulator-only development trials on compute3."""
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

B = Path('/home/physicalai/phantom-icra-2027/sim/waffles')
R = B / 'runs/teacher_success_anchor_v7'
D = B / 'source_teacher_anchor_driver_v7'
S = B / 'source_teacher_anchor_rpc_v7'
LIVE = Path('/home/physicalai/phantom-icra-2027/phantom')
PYTHON = LIVE / '.venv/bin/python'
CAMPAIGN = D / 'configs/sim/teacher_success_anchor_v7_diagnostic.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, indent=2)
        stream.write('\n')


def main():
    assert not (R / 'diagnostic').exists()
    assert not (R / 'launch.json').exists()
    assert json.loads((B / 'runs/teacher_success_anchor_v5/transition/final_status.json').read_text())['status'] == 'study_complete'
    assert json.loads((B / 'runs/teacher_success_anchor_v6/handoff/final_status.json').read_text())['status'] == 'diagnostic_driver_complete'
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 7799))
    assert sha(CAMPAIGN) == 'e68ca523862e805538b9f150c8394363091aa9d9dac6bd950b6215fc148d739f'
    source_manifest = R / 'source_audit/source_manifest.json'
    assert sha(source_manifest) == '1165cb34c53444c89ea76a78c9687e508adfd70b22570c2278060e83afd37084'
    source_files = json.loads(source_manifest.read_text())['output_sha256']
    for name, expected in source_files.items():
        assert sha(S / name) == expected, name
    driver_files = {str(p.relative_to(D)): sha(p) for p in sorted(D.rglob('*'))
                    if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    assert driver_files['tools/sim/run_policy_campaign.py'] == '235db3d485412da413c9abf866d22c0ff63522fdba70de65b6f7d5234b79aefa'
    assert driver_files['tools/sim/analyze_policy_campaign.py'] == '11b88baedfa6bed8696abb87fb30a51677be56538267c9fef3ddc60743d307a4'
    write(R / 'external_driver_manifest.json', {'source': str(D), 'sha256': driver_files})
    command = [str(PYTHON), str(D / 'tools/sim/run_policy_campaign.py'),
               '--campaign', str(CAMPAIGN), '--source', str(S), '--live-repo', str(LIVE),
               '--server-python', str(PYTHON), '--evidence', str(B / 'evidence'),
               '--hardware-config', str(B / 'runs/teacher_pick_place_v1/runtime/hardware_campaign.yaml'),
               '--robot-usd', str(B / 'runs/validation_v2/pick_place/robot_asset/waffles/waffles.usda'),
               '--output', str(R / 'diagnostic'), '--port', '7799']
    sys.path.insert(0, str(D))
    spec = importlib.util.spec_from_file_location('v7_driver', D / 'tools/sim/run_policy_campaign.py')
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    args = driver.parser().parse_args(command[2:])
    design, design_sha = driver.load_design(CAMPAIGN)
    frozen = driver.source_manifest(args, design_sha, design)
    assert frozen['hardware_sha256'] == design['runtime_hardware']['sha256']
    assert sha(LIVE / 'runs/teacher_v5_ftA/teacher_001500.pt') == design['policies'][0]['checkpoint_sha256']
    assert design['adapter_profile']['policy_delivery_clock'] == 'rpc_wall'
    assert design['adapter_profile']['servo_reach_limiter'] is False
    assert sum(len(block['trials']) for block in driver.blocks(design)) == 4
    write(R / 'preflight_inputs.json', frozen)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    with (R / 'diagnostic.log').open('x') as log:
        process = subprocess.Popen(command + ['--execute'], cwd=D, env=env,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    launch = {'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'pid': process.pid, 'command': command + ['--execute'],
              'campaign_sha256': design_sha, 'source_file_count': len(source_files),
              'driver_file_count': len(driver_files), 'launcher_sha256': sha(Path(__file__)),
              'planned_trials': 4, 'hardware_actions': False, 'automatic_expansion': False}
    write(R / 'launch.json', launch)
    print(json.dumps(launch, indent=2))


if __name__ == '__main__':
    main()
