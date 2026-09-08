#!/usr/bin/env python3
"""Freeze the reviewed V8 six-case diagnostic; no process or inference launch."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

B=Path('/home/physicalai/phantom-icra-2027/sim/waffles')
R=B/'runs/teacher_carry_hotfix_v8'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def inventory(p):return {str(f.relative_to(p)):sha(f)for f in sorted(p.rglob('*')) if f.is_file() and '__pycache__' not in f.parts and f.suffix!='.pyc'}
assert not (R/'bindings.frozen.json').exists() and not (R/'paired').exists()
log=R/'cpu_assembled_tests.log'
assert '100 passed' in log.read_text() and 'failed' not in log.read_text()
now=datetime.now(timezone.utc).isoformat()
for name,key in [('runtime_manifest','output_sha256'),('driver_manifest','sha256'),('inference_manifest','sha256')]:
 p=R/(name+'.json');d=read(p);values=inventory(Path(d['root']))
 assert values==d[key], (name,'unexpected source change since reviewed inventory')
 write(R/(name+'.draft.json'),d)
 d.update(status='frozen',frozen_at_utc=now,review='Root approved after assembled100tests passed; noGPUlaunch')
 write(p,d)
for p in (R/'campaigns').glob('*.json'):
 d=read(p);assert d['status']=='draft_pending_root_freeze'
 write(R/'campaigns'/(p.stem+'.draft.json'),d)
 d.update(status='frozen',frozen_at_utc=now,freeze_authorization='Root approved final tested source; six matched development cells only')
 write(p,d)
b=read(R/'bindings.draft.json')
b.update(status='frozen',cpu_tests_passed=True,root_review_approved=True,frozen_at_utc=now,interpretation='Six matched development cells; no automatic expansion or reliable-winner claim')
for row in b['files']:row['sha256']=sha(Path(row['path']))
for row in b['inventories']:row['sha256']=sha(Path(row['manifest']))
for row in b['cases']:row['campaign_sha256']=sha(Path(row['campaign']))
testmeta={'status':'passed','tests':100,'log':str(log),'log_sha256':sha(log),'test_fixture_sha256':inventory(R/'cpu_suite'),'runtime_manifest_sha256':sha(R/'runtime_manifest.json'),'execution':'CPU mocks only; no model/Isaac/hardware construction'}
write(R/'cpu_assembled_tests.json',testmeta)
for p in [log,R/'cpu_assembled_tests.json',R/'inference_paths_preflight.json']:
 b['files'].append({'path':str(p),'sha256':sha(p)})
write(R/'bindings.frozen.json',b)
print(json.dumps({'status':'frozen','binding_sha256':sha(R/'bindings.frozen.json'),'source_inventories':b['inventories'],'campaigns':b['cases']},indent=2))
