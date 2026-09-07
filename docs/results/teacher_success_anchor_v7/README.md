# Complete-client timing and K1/K4 diagnostic

This separate four-case development study uses the leading ftA1500 teacher, EMA/NFE1/guidance1, with K4 and K1 under explicit `policy_delivery_clock=rpc_wall`. Both use the unchanged successful scene/start, minimal gel-v2/FINISH profile, original hardware bounds, and disabled servo limiter. Previously observed development seeds904301/904302 run in K4/K1 then K1/K4 order. These are not reserved model-selection cases.

The [campaign](rpc_k_diagnostic_campaign.json) was frozen before execution, SHA256 `e68ca523862e805538b9f150c8394363091aa9d9dac6bd950b6215fc148d739f`. A prelaunch draft was cleaned of inherited V5 protocol/source bindings before any plan or model execution; its hash/reason are retained in the campaign. The final source, driver, inputs and launch are pinned in [source manifest](source_manifest.json), [driver manifest](external_driver_manifest.json), [input preflight](preflight_inputs.json) and [launch ledger](launch.json). All4351 source files and345 external-driver files were checked before starting.

Two stale parent-study sentences remained in `comparability_requirements`; the [prose erratum](campaign_prose_erratum.md) discloses them without modifying the frozen campaign. The explicit executable design is two recipes/four development trials, not a four-model prospective ranking.

Complete-client delivery measures `policy.replan` after observation construction and before audit/bookkeeping. Native Plan latency, action grid and CPK token remain unchanged. Existing executor submission skips/rebases samples already expired when a response arrives. The source audit and actual-adapter CPU checks establish these semantics; physics/model behavior is evaluated by these separate trials. K1 changes candidate selection and random draw shape as well as compute, so it is not a pure timing intervention.

**Completed: four valid cases, independently rescored with exact agreement and4,859 source/driver/input checks. Retain K4.** No policy failure was retried and no additional study was launched.

| Development settings | Acquired | Lift/carry | Strict placement / clean FINISH | Drops | Terminal outcome |
|---|---:|---:|---:|---:|---|
| NFE1 / K4 |2/2|2/2|1/2|0/2|One60s clean hold, one wrist-extension stop|
| NFE1 / K1 |2/2|0/2|0/2|0/2|One wrist-extension stop, one60s horizon without lift|

K1 reduced the mean of per-episode full-client durations to0.27933s versus0.84776s for K4, but did not improve physical progression. Mean native durations were0.27620s and0.84465s respectively. The roughly3.1ms client-minus-native difference is distinct from outer observation/logging overhead. This diagnostic supports keeping K4; it does not establish reliable placement, hardware transfer or a causal benefit from the timing correction. Two reused development seeds cannot establish a reliable winner and are not pooled into V5 confirmation. No fresh settings confirmation or arm extension was triggered.

The [independent audit](four_case_independent_audit.md) gives all four physical events, stops, force diagnostics and timing, backed by [machine-readable checks](four_case_independent_audit.json) and the [completed primary summary](diagnostic/summary.json). K4 seed904301 placed at24.000s, reached FINISH at24.172s and held through60s. The other K4 case carried before wrist extension; both K1 cases failed to lift. Peak pad-packet forces are uncalibrated diagnostics, not hardware safety limits.

The [timing audit](timing_audit.md) verifies all296 returned plan-clock contracts and distinguishes native inference, client duration and outer runner cost. All four tactile videos are indexed in the [video manifest](diagnostic_video_manifest.json); [final checks](diagnostic_video_final_audit.json) confirm2,307 decoded frames,7,074,458 bytes and matching causal sensor/score hashes. The owned simulator, server and renderer processes have exited.

Remote runtime: `/home/physicalai/phantom-icra-2027/sim/waffles/source_teacher_anchor_rpc_v7`; driver: adjacent `source_teacher_anchor_driver_v7`; outputs: `runs/teacher_success_anchor_v7/diagnostic`. The [launcher](launch_diagnostic.py) contains the exact simulator command and refuses existing output or a busy owned port. It was run once after V5 and V6 completed. Reproduction requires fresh output and the pinned sources/inputs; do not modify existing raw trials.

The compact runtime and driver archives are under the remote V7 `reproduction_archives/` directory, with member hashes in [archive manifest](reproduction_archives_manifest.json). They omit weights, recordings and the Isaac installation. No hardware was launched or altered.
