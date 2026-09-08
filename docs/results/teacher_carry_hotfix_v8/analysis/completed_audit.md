# V8 paired carry-hold audit

Six contemporaneous development trials, two seeds per setting. Physical outcomes come from the unchanged authoritative strict scorer. No historical pooling or causal estimate from unmatched inference timing.

| Setting / seed | Acquire / lift / carry / full / drop | Stop | End | Max measured radius | Holds / max elapsed |
|---|---|---|---:|---:|---|
| plain_k4 / 904301 | 1/1/1/0/0 | safety_stop | 17.616 s | 468.04 mm | 0 / 0.000 s |
| plain_k4 / 904302 | 1/0/0/0/0 | safety_stop | 17.672 s | 468.01 mm | 0 / 0.000 s |
| legacy_limiter / 904301 | 1/1/1/0/0 | servo_limiter_stall | 17.104 s | 461.53 mm | 0 / 0.000 s |
| legacy_limiter / 904302 | 1/0/0/0/0 | servo_limiter_stall | 17.360 s | 461.58 mm | 0 / 0.000 s |
| bounded_hold / 904301 | 1/1/1/0/0 | none | 59.996 s | 461.58 mm | 169 / 1.264 s |
| bounded_hold / 904302 | 1/0/0/0/0 | safety_stop | 44.480 s | 461.59 mm | 276 / 1.704 s |

Audit status: **audit_pass**. [Numerical audit](completed_audit.json) includes frozen hashes, fixed thresholds, physical support, command/FK checks, independent hold-budget replay and post-hold plan delivery/activation.

FINISH is not physical success. A hold timeout is an executed controller outcome, not an infrastructure failure. Constraint and joint limits bound submitted commands; they do not certify measured stopping distance, contact-force realism or hardware safety. Fresh capture/delivery timestamps show whether a response could play; they do not establish that it recovers the task.
