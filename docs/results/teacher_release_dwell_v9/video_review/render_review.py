#!/usr/bin/env python3
"""Render and verify the four completed V9 runs; never start simulator trials."""

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

BASE = Path('/home/physicalai/phantom-icra-2027/sim/waffles')
ROOT = BASE / 'runs/teacher_release_dwell_v9'
SOURCE = BASE / 'source_teacher_carry_hotfix_v8'
PYTHON = '/home/physicalai/phantom-icra-2027/phantom/.venv/bin/python'
RENDERER = ROOT / 'review_tools/make_policy_video.py'
OUTPUT = ROOT / 'video_reviews'
SETTINGS = {'control_dwell200': 0.2, 'treatment_dwell100': 0.1}
CASES = [(setting, seed) for setting in SETTINGS for seed in [904301, 904302]]


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()



def completion_guard():
    """Require completed scores and independently confirmed cleanup before encoding."""
    complete = json.loads((ROOT / 'paired/complete.json').read_text())
    assert complete['status'] == 'four_valid_trials_completed', complete
    assert complete['trials'] == 4, complete
    audit = json.loads((ROOT / 'completion_audit.json').read_text())
    assert audit['status'] == 'passed', audit
    assert audit['planned'] == 4 and audit['valid'] == 4, audit
    for key in ['all_trials_complete', 'owned_processes_stopped', 'dedicated_port_free']:
        assert audit[key] is True, (key, audit.get(key))
    plans = json.loads((ROOT / 'paired/launch_plan.json').read_text())['plans']
    assert [p['id'] for p in plans] == list(SETTINGS)
    for setting, seed in CASES:
        block = ROOT / 'paired' / setting
        progress = json.loads((block / 'progress.json').read_text())
        assert progress['status'] == 'all_trials_completed'
        assert len(progress['trials']) == 2
        assert all(row['status'] == 'completed' for row in progress['trials'].values())
        case = f'fta1500_nfe1_k4__successful_anchor__seed{seed}'
        score = json.loads((block / 'analysis/trials' / f'{case}.json').read_text())
        assert score['metrics']['valid_for_scoring'] is True
        assert score['sampling_seed'] == seed
        info = json.loads((block / 'rollouts' / case / 'policy_info.json').read_text())
        expected_dwell = SETTINGS[setting]
        assert info['placement_release']['opening_hold_s'] == expected_dwell


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    completion_guard()
    if args.check_only:
        print(json.dumps({'status': 'all_four_complete_and_cleanup_confirmed'}))
        return
    import cv2

    environment = dict(os.environ, PHANTOM_REVIEW_SOURCE=str(SOURCE),
                       PYTHONDONTWRITEBYTECODE='1')
    manifest = {'schema_version': 1, 'campaign': str(ROOT),
                'renderer_sha256': sha256(RENDERER), 'runtime_source': str(SOURCE),
                'new_trials_started': 0, 'videos': [],
                'completion_audit_sha256': sha256(ROOT / 'completion_audit.json'),
                'launch_plan_sha256': sha256(ROOT / 'paired/launch_plan.json')}
    for setting, seed in CASES:
        case = f'fta1500_nfe1_k4__successful_anchor__seed{seed}'
        run = ROOT / 'paired' / setting / 'rollouts' / case
        score_file = ROOT / 'paired' / setting / 'analysis/trials' / f'{case}.json'
        score = json.loads(score_file.read_text())
        directory = OUTPUT / setting / case
        directory.mkdir(parents=True, exist_ok=True)
        output = directory / 'policy_review.mp4'
        command = [PYTHON, str(RENDERER), '--run', str(run), '--output', str(output),
                   '--label', f'{setting} | seed {seed} | teacher ftA1500']
        if not output.exists():
            result = subprocess.run(command, env=environment, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            (directory / 'render.log').write_text(result.stdout)
            result.check_returncode()
        metadata = json.loads(output.with_suffix('.json').read_text())
        assert metadata['trials'][0]['metrics'] == score['metrics']
        assert metadata['trials'][0]['tactile_mapping']['actual_input_proxy']
        assert metadata['trials'][0]['tactile_mapping']['calibrated'] is False
        for row in metadata['trials'][0]['frame_mapping']:
            assert row['scene_age_s'] is None or row['scene_age_s'] >= -1e-8
            assert row['runtime_tactile_age_s'] is None or row['runtime_tactile_age_s'] >= -1e-8
        subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(output),
                        '-f', 'null', '-'], check=True)
        cap = cv2.VideoCapture(str(output))
        assert cap.isOpened()
        n = 0
        last = None
        snapshots = []
        times = [8, 12, 15, 22, 30, 45]
        wanted = {round(t * metadata['fps']): t for t in times
                  if t <= metadata['horizon_s']}
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if n in wanted:
                frame_path = directory / f'frame_{wanted[n]:05.2f}s.png'
                assert cv2.imwrite(str(frame_path), frame)
                snapshots.append({'path': str(frame_path), 'frame': n,
                                  't_s': n / metadata['fps'],
                                  'sha256': sha256(frame_path)})
            last = frame
            n += 1
        cap.release()
        assert n == metadata['frames'], (n, metadata['frames'])
        assert last is not None
        final_path = directory / 'frame_final.png'
        assert cv2.imwrite(str(final_path), last)
        snapshots.append({'path': str(final_path), 'frame': n - 1,
                          't_s': (n - 1) / metadata['fps'],
                          'sha256': sha256(final_path)})
        record = {'setting': setting, 'seed': seed, 'run': str(run),
                  'video': str(output), 'video_sha256': sha256(output),
                  'bytes': output.stat().st_size,
                  'metadata_sha256': sha256(output.with_suffix('.json')),
                  'authoritative_score_sha256': sha256(score_file),
                  'input_sha256': {name: sha256(run / name) for name in
                                   ['sim.mp4', 'policy_tactile.npz', 'sim_trace.npz']},
                  'fps': metadata['fps'], 'decoded_frames': n,
                  'horizon_s': metadata['horizon_s'],
                  'ffmpeg_full_decode': 'passed',
                  'opencv_full_decode': 'passed',
                  'metric_equality_to_frozen_score': True,
                  'causal_scene_and_tactile_alignment': True,
                  'tactile_mapping': metadata['trials'][0]['tactile_mapping'],
                  'outcomes': score['metrics']['outcomes'],
                  'object': score['metrics']['object'],
                  'placement_support': score['metrics']['placement_support'],
                  'snapshots': snapshots}
        manifest['videos'].append(record)
        manifest['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
        (OUTPUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print(json.dumps({key: record[key] for key in
                          ['setting', 'seed', 'video_sha256', 'decoded_frames']}), flush=True)
    manifest['status'] = 'four_videos_rendered_and_fully_decoded'
    (OUTPUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
