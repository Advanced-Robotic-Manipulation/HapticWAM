#!/usr/bin/env python3
"""Read-only process/case monitoring; writes separate CPU monitor scores only."""
import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
B = Path('/home/physicalai/phantom-icra-2027/sim/waffles')
R = B/'runs/teacher_carry_hotfix_v8'
D = B/'source_teacher_carry_hotfix_driver_v8'
sys.path.insert(0,str(D))
spec=importlib.util.spec_from_file_location('v8analyzer',D/'tools/sim/analyze_policy_campaign.py')
an=importlib.util.module_from_spec(spec);sys.modules[spec.name]=an;spec.loader.exec_module(an)
seen_path=R/'monitor_seen.json'
seen=json.loads(seen_path.read_text()) if seen_path.exists() else []
result={'launcher_alive':Path('/proc/3649515').exists(),'completed':[],'new':[],'status':{},'errors':[]}
for name in ['plain_k4','legacy_limiter','bounded_hold']:
 root=R/'paired'/name
 p=root/'progress.json'
 if not p.exists():continue
 progress=json.loads(p.read_text());result['status'][name]=progress['status']
 completed=[k for k,v in progress['trials'].items() if v['status']=='completed']
 result['completed'] += [name+'/'+k for k in completed]
 if any(name+'/'+k not in seen for k in completed):
  design,digest=an.load_design(R/'campaigns'/f'{name}.json')
  records=an.load_trials(root,design,digest,R/'monitor_analysis'/name)
  for rec in records.values():
   key=name+'/'+Path(rec['directory']).name
   if key not in result['completed'] or key in seen:continue
   m=rec.get('metrics') or {}
   run=json.loads((Path(rec['directory'])/'run.json').read_text())
   result['new'].append({'case':key,'status':rec['status'],'valid':m.get('valid_for_scoring'),'invalid_reasons':m.get('invalid_reasons'),'outcomes':m.get('outcomes'),'events':m.get('event_times_s'),'object':m.get('object'),'control':m.get('control'),'run_stop_reason':run.get('stop_reason'),'run_safety_events':run.get('safety_events'),'duration_s':run.get('duration_s')})
   seen.append(key)
 if progress['status'] not in ('running','all_trials_completed'):
  result['errors'].append({'condition':name,'progress':progress})
seen_path.write_text(json.dumps(seen,indent=2)+'\n')
(R/'monitor_latest.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(result,indent=2,allow_nan=False))
