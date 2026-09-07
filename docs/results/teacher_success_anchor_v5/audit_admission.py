#!/usr/bin/env python3
"""CPU source/input admission audit; never gates comparison on teacher success."""

import hashlib
import json
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DOC = Path(__file__).parent
campaign = ROOT / "configs/sim/teacher_success_anchor_v5_screen.json"
design = json.loads(campaign.read_text())
script = """
import hashlib,json,sys
from pathlib import Path
specs=json.load(sys.stdin)
rows=[]
for r in specs:
 h=hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()
 rows.append({**r,'actual_sha256':h,'match':h==r['sha256']})
print(json.dumps(rows))
"""
result = subprocess.run(
    ["ssh", "compute3", "python3", "-c", shlex.quote(script)],
    input=json.dumps(design["runtime_contract"]["source_and_input_hashes"]),
    text=True,
    capture_output=True,
    check=True,
)
checks = json.loads(result.stdout)
mechanics = ROOT / design["execution_admission"]["mechanics_audit"]["path"]
mechanics_sha = hashlib.sha256(mechanics.read_bytes()).hexdigest()
mechanics_ok = (
    mechanics_sha == design["execution_admission"]["mechanics_audit"]["sha256"]
    and json.loads(mechanics.read_text())["status"] == "passed"
)
audit = {
    "status": "passed"
    if all(c["match"] for c in checks) and mechanics_ok
    else "failed",
    "campaign_sha256": hashlib.sha256(campaign.read_bytes()).hexdigest(),
    "protocol_sha256": design["protocol"]["sha256"],
    "source": design["runtime_contract"]["source"],
    "source_manifest": design["runtime_contract"]["source_manifest"],
    "checks": checks,
    "mechanics_audit_sha256": mechanics_sha,
    "mechanics_integrity_passed": mechanics_ok,
    "reference_teacher_success_required": False,
    "policy_outcomes_read": False,
    "semantics": "Verify120 immutable source/native/input files, including all four declared replacements. Bytecode caches are intentionally excluded. Historical mechanics and saved-input inference validation admit comparison; minimal diagnostic policy success is not a gate.",
    "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
out = DOC / "admission_audit.json"
with out.open("x") as f:
    f.write(json.dumps(audit, indent=2) + "\n")
print(json.dumps({"status": audit["status"], "checks": len(checks), "out": str(out)}))
