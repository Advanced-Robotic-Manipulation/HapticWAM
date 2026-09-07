# Opt-in native minimal-v5 controller port

The local native port passes **32 focused CPU checks**, including exact equality
for all **37 archived delivered action arrays** from the successful seed-4242
anchor. It has not been installed on the live rig, executed on hardware, or used
to change frozen simulator results. This is controller-port qualification;
physical sensor transfer, geometry and asynchronous I/O still need bench checks.
Full evidence and source hashes are in
[native_controller_port_audit.json](native_controller_port_audit.json).

The selector is `--placement-controller-profile minimal_v5` on native
`phantom.scripts.run_deploy`. It is absent by default. Explicit selection requires
teacher mode, `--terminal-veto`, an explicitly supplied release JSON with measured
positive TCP bounds, `finish_after_release: true`, and the existing enabled load
latch. The existing release validator also rejects a reclose threshold above
hardware maximum closure; the profile rejects a recovery aperture outside
`[0, hw.gripper.max_close_cmd]`, including malformed task-start statistics.
These checks happen before the native driver factory;
the CLI performs its profile check before model loading. This document does not
provide or execute a hardware control invocation.

The profile selects a small historical-veto mixin from
[phantom/deploy/minimal_v5.py](../../../phantom/deploy/minimal_v5.py). The historical
method's AST matches the immutable v1 bridge exactly. The only planner-loop
change is a profile predicate: this opt-in uses request-snapshot TCP/aperture for
veto, while default current-native release keeps delivery feedback. Model
conditioning remains the original snapshot in both paths. Release permission
still uses current measured TCP inside the reviewed physical volume.

Historical close masking, floor-only exclusion and bounded recovery are
preserved, including its absence of the later live tactile phantom recovery and
loaded-pad recovery exclusion. Historical recovery does not clear the running
load latch. This is an explicit controller choice, not a claim that it includes
every newer optional planner behavior. The unchanged SafetyMonitor, governor and
real-driver guards remain authoritative.

Original teacher opening samples can pass the whole-chunk close mask only while
the release window is active. Arm-channel recovery edits remain in force and
rewritten contact-package feedback is invalidated. Release requires the prior
loaded latch and sustained policy opening, followed by measured bilateral
unload/open dwell. FINISH additionally requires an accepted open command and
measured TCP that does not require a clamp. It holds the achieved measured pose,
prevents further model plans and keeps safety/sensor observation active.

The existing native gripper mailbox has extra physical timing constraints:
FINISH waits for its nonblocking I/O lock and actually acknowledged open command;
stale gripper feedback cannot prove completion. Native post-FINISH observation
uses explicit `finish_observation_s` (default 2 s). The simulator instead keeps
its common 60 s physics horizon. The parity tests use ideal acknowledged mailbox
timing and do not claim identical wall-clock trajectories on the real robot.

The executor, SafetyMonitor, governor, shared release controller, simulator
adapter, inference implementation and hardware schema are byte-identical to
the pinned pre-port HEAD in the audit. This audit describes only the controller
port and does not qualify a separately proposed reach-limiter extraction.
The default planner class identity and existing default metadata
are preserved. Real tactile/wrist/RGB streams, checkpoint normalizers and
preprocessing remain native; no gel image baseline or PhysX wrist proxy is
ported. No hardware safety bound or force threshold is relaxed.

The test evidence includes saved-plan masks/recovery/opening passthrough and CPK
invalidation; request-versus-delivery feedback with an unchanged captured model
observation; synthetic pickup, retained latch, release and FINISH parity against
the hash-pinned minimal adapter; missing/invalid opt-in rejection before a fake
driver factory; finite-input rejection; accepted-open/stale/mailbox safeguards;
and safety or operator/sensor interruption before and after completion. No model
or GPU was loaded. Reproduce those checks from the repository with:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/test_native_minimal_v5.py tests/test_sim_release_finish.py \
  tests/test_sim_teacher_anchor_finish.py -q
```

Native episode metadata records `deploy_overrides.placement_controller_profile`
with the profile ID, historical and port source hashes, effective veto/release
settings and timing limitations. `placement_veto_feedback` records
`request_snapshot_historical`; planner records retain the original proposals
when rewritten and annotate `controller_profile`, `implementation` and
`feedback_source`. Completion is separately retained in `stop.json` and runtime
release diagnostics. Neither completion nor a clean controller stop establishes
that the object is settled in the measured box.
