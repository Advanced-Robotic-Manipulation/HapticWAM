"""Read-only audit of saved teacher observations and physical contact evidence."""
import hashlib
import io
import json
from pathlib import Path

import numpy as np

BASE = Path('/home/physicalai/phantom-icra-2027/sim/waffles')
OUT = BASE / 'runs/teacher_behavior_debug_20260907/sensors'


def load_json(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(rows):
    held = [r for r in rows if r['bilateral_packet_contact_at_sensor_time']]
    return {
        'snapshots': len(rows), 'bilateral_contact_snapshots': len(held),
        'bilateral_with_no_added_gel_force_either_pad': sum(max(r['gel_normal_added_n']) < .1 for r in held),
        'bilateral_with_one_or_more_pads_below_gel_0_1n': sum(min(r['gel_normal_added_n']) < .1 for r in held),
        'bilateral_with_one_or_more_masks_below_training_contact_threshold': sum(min(r['contact_mask_fraction']) <= .025 for r in held),
        'bilateral_with_predicted_hold_below_0_5': sum(r['predicted_events'][2] < .5 for r in held),
        'reactive_exact_zero': sum(r['reactive'] == 0 for r in rows),
        'reactive_median': float(np.median([r['reactive'] for r in rows])) if rows else None,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    baseline_path = BASE / 'runs/teacher_pick_place_v1/sept4_no_contact_sensor_baseline.npz'
    with np.load(baseline_path) as z:
        base_gel = np.stack([z[f'{side}_infer_img'] for side in ['left', 'right']]).astype(np.float32)
        base_wrench = np.stack([z[f'{side}_wrench'] for side in ['left', 'right']])
    groups = [('v5_confirmation', BASE / 'runs/teacher_success_anchor_v5/confirmation'),
              ('v7_development', BASE / 'runs/teacher_success_anchor_v7/diagnostic')]
    result = {'scope': 'Saved-input read-only diagnostic; no inference, mutation, rescoring, or causal model attribution.',
              'alignment': 'Physical contacts sampled causally at the saved tactile timestamp on a15Hz trace; quantization up to one trace interval. No temporal interpolation.',
              'baseline_sha256': digest(baseline_path), 'helper_sha256': digest(Path(__file__)), 'cases': []}
    for group, root in groups:
        for path in sorted((root / 'rollouts').glob('fta1500_*')):
            score_path = root / 'analysis/trials' / (path.name + '.json')
            score = load_json(score_path)
            assert score['metrics']['valid_for_scoring'], path.name
            plans_path = path / 'planner_trace.json'
            plans = {r['replan_id']: r for r in load_json(plans_path) if 'actions' in r}
            trace_path = path / 'sim_trace.npz'
            with np.load(trace_path) as z:
                trace_t = z['t'].copy()
                forces = z['pad_packet_normal_force'].copy()
                obj = z['waffle_position'].copy()
            rows, hashes = [], {}
            for observation in sorted((path / 'observations').glob('*.npz')):
                raw = observation.read_bytes()
                hashes[observation.name] = hashlib.sha256(raw).hexdigest()
                meta = load_json(observation.with_suffix('.json'))
                replan = plans[meta['replan_id']]
                with np.load(io.BytesIO(raw), allow_pickle=False) as z:
                    t = float(z['t'])
                    cs = z['contact_state'].astype(np.float64)
                    fields = z['fields'].astype(np.float64)
                    gel = z['gel'].astype(np.float32)
                    arm = z['ur_state'].astype(np.float64)
                    wrist = z['wrist_window'].astype(np.float64)
                    tactile_t = float(min(meta['sensor_times']['tactile_left'], meta['sensor_times']['tactile_right']))
                    idx = int(np.clip(np.searchsorted(trace_t, tactile_t, side='right')-1, 0, len(trace_t)-1))
                    delta_force = cs[:, :6] - base_wrench
                    row = {'replan_id': meta['replan_id'], 't': t, 'tactile_t': tactile_t,
                           'physical_trace_t': float(trace_t[idx]), 'bilateral_packet_contact_at_sensor_time': bool(np.all(forces[idx] > .1)),
                           'physical_packet_normal_force_n': forces[idx].tolist(),
                           'physical_packet_rise_m': float(obj[idx,2]-obj[0,2]),
                           'gel_normal_added_n': (-delta_force[:,2]).tolist(),
                           'contact_mask_fraction': cs[:,10].tolist(), 'contact_area_sdk': cs[:,6].tolist(),
                           'contact_slip': cs[:,9].tolist(), 'contact_cop': cs[:,7:9].tolist(),
                           'contact_wrench_sdk': cs[:,:6].tolist(),
                           'field_depth_peak_mm': np.max(fields[:,:,:,2], axis=(1,2)).tolist(),
                           'field_integral_force_sdk': (fields[:,:,:,5:8].mean(axis=(1,2))*110592).tolist(),
                           'gel_rmse_from_static_baseline_uint8': np.sqrt(((gel-base_gel)**2).mean(axis=(1,2))).tolist(),
                           'reactive': float(z['reactive']), 'predicted_events': replan['p_evt'],
                           'predicted_head_displacement_m': np.asarray(replan['actions'])[:10,:3].sum(axis=0).tolist(),
                           'predicted_head_gripper_mean': float(np.asarray(replan['actions'])[:10,6].mean()),
                           'tcp': arm[12:18].tolist(), 'wrist_mean_raw': wrist.mean(axis=0).tolist()}
                    assert all(np.isfinite(v).all() for v in [cs,fields,gel,arm,wrist])
                    rows.append(row)
            events = score['metrics']['event_times_s']
            phases = {'all': rows}
            for label,lo,hi in [('after_acquisition_before_lift',events.get('acquisition'),events.get('lift')),
                                ('after_lift_before_release',events.get('lift'),events.get('release_in_bin'))]:
                phases[label] = [] if lo is None else [r for r in rows if r['t'] >= lo and (hi is None or r['t'] < hi)]
            case = {'group': group,'case':path.name,'outcomes':score['metrics']['outcomes'],'events':events,
                    'input_sha256': {'score':digest(score_path),'planner':digest(plans_path),'trace':digest(trace_path),'observations':hashes},
                    'phase_summaries':{k:summary(v) for k,v in phases.items()},'observations':rows}
            result['cases'].append(case)
            print(group,path.name,case['phase_summaries'],flush=True)
    (OUT/'observation_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('wrote',OUT/'observation_audit.json',flush=True)


if __name__ == '__main__':
    main()
