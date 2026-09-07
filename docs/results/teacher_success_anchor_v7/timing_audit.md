# Four-case RPC timing and outcome audit

All **four planned development trials** completed and are valid. The read-only
[numeric audit](timing_audit.json) checks every returned plan against the actual
client-clock contract and preserves per-case input/score hashes. Its
[CPU helper](audit_timing.py) does not perform inference, change scores or modify
recordings. These reused development seeds are not fresh ranking confirmation.

| Recipe | Seed | Physical result | FINISH | Actual stop | Last telemetry |
| --- | --- | --- | --- | --- | --- |
| NFE1/K4 | 904301 | Acquired, lifted, carried; strict supported placement at 24.000 s | 24.172 s | None | 59.996 s |
| NFE1/K4 | 904302 | Acquired, lifted and carried; no release | None | Wrist extension, 16.236 s | 18.232 s |
| NFE1/K1 | 904301 | Acquired; no lift or placement | None | Wrist extension, 13.516 s | 15.512 s |
| NFE1/K1 | 904302 | Acquired; no lift or placement | None | None; horizon reached | 59.996 s |

The successful K4 case commits policy-commanded release at 23.300 s after
sustained opening from 23.100 s. Physical release-in-bin is sampled at 23.336 s,
and strict supported placement is independently confirmed at 24.000 s. Measured
unloaded/open dwell begins at 23.972 s; the separate controller FINISH follows
at 24.172 s. Its achieved-TCP reference remains constant for 4,479 execution
rows through 59.996 s, with no subsequent replan or stop. Final packet position
is approximately `[-0.419509, 0.067694, 0.072000]` m, with 0.34393 N of simulated
bin support and zero packet–robot normal contact throughout the post-FINISH
period. These contact values are simulator evidence, not calibrated hardware
force measurements. FINISH itself never supplies object-state success.

All **296 returned plans** retain the native action grid and native
`Plan.latency_s`; actual delivery uses measured client-call wall time. Every
duration is finite, client duration covers native duration, and all accepted
activation times follow their scheduled delivery. The maximum additional
executor quantization is 7.925 ms, within the 8 ms control period. There are no
inference errors, IK rejections, explicit stale holds or separate playback-cap
dwell in any of these four cases. Existing physical-score definitions are
unchanged.

| Recipe | Returned calls | Median native inference | Median client call | Median client minus native | Median outer runner minus client |
| --- | ---: | ---: | ---: | ---: | ---: |
| NFE1/K4 | 47 | 0.792147 s | 0.795359 s | 2.648 ms | 164.734 ms |
| NFE1/K1 | 249 | 0.215887 s | 0.218694 s | 2.544 ms | 194.469 ms |

These are pooled call-level descriptive statistics from two episodes per
recipe, with unequal call counts. Medians of per-call differences are computed
directly, rather than subtracting the two marginal medians. K1 is faster in
these observations, but neither K1 trial lifts or places. No causal benefit of
RPC delivery, model-ranking winner or hardware success rate follows from this
four-case comparison.

**Correction to the earlier approximately 0.2 s RPC-overhead attribution:** the
old `inference_wall_time_s` records the outer runner path, including snapshot
preparation, observation-saving callbacks and adapter bookkeeping. It does not
isolate RPC transport. The directly measured complete-client excess over native
inference is only a few milliseconds here; the much larger excess lies outside
the policy call. Client-minus-native also includes remaining policy/client work,
so even that difference is not a pure network timer. Historical raw numbers and
frozen results remain unchanged. Legacy score field
`control.effective_latency_s` still reports native `Plan.latency_s`; the explicit
client/delivery fields in this audit disambiguate it.

The [source review](rpc_independent_review.md) records the remaining limits:
physics is paused during the synchronous call and advances afterward; native
CPK/selection still use native timing; historical request-time veto, pad-only
wrist feedback and uncalibrated gel mapping remain. K1 removes beam selection
and changes random draws. Both recipes use the same ftA1500 EMA checkpoint,
NFE 1, guidance 1, parity/persistent noise, task `waffles`, max-play 10,
standard v5 safety configuration and limiter disabled. Their order is K4/K1 for
904301, then K1/K4 for 904302.

All four [review videos](diagnostic_video_manifest.json) passed complete decode,
exact saved-gel and causal-timestamp checks, and raw-input/primary-score hash
verification. The [final presentation audit](diagnostic_video_final_audit.json)
records 2,307 decoded frames, 7,074,458 video bytes and owned watcher PID 2564015
exited. Each video covers its actual telemetry duration, including post-stop
physics, without padding shorter trials to 60 s. Videos remain under remote
`runs/teacher_success_anchor_v7/video_reviews/diagnostic/<case>/policy_review.mp4`.
