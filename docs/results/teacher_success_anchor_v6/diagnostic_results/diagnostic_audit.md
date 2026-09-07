# V6 limiter diagnostic comparison

Descriptive development comparison; native timing differs and physical success is scored independently of FINISH.

| Group / seed | Acquired / lift / carry / full | Termination | Limiter modes | Max command speed | Max measured radius |
|---|---|---|---|---:|---:|
| minimal_baseline / 904301 | 1/1/1/0 | measured_safety_stop | {} | 1.000000 rad/s | 0.468094 m |
| minimal_baseline / 904302 | 1/0/0/0 | measured_safety_stop | {} | 1.000000 rad/s | 0.468052 m |
| limiter_v6 / 904301 | 1/1/1/0 | executed_limiter_stall | {'unchanged': 1552, 'step': 54, 'slide': 3, 'hold': 25} | 0.996181 rad/s | 0.461548 m |
| limiter_v6 / 904302 | 1/1/1/0 | executed_limiter_stall | {'unchanged': 1652, 'step': 95, 'slide': 1, 'hold': 25} | 0.999416 rad/s | 0.461545 m |

Audit status: **audit_pass**. Details, raw hashes, strict metrics, startup branch checks and distinct commanded/measured fields are in [diagnostic_audit.json](diagnostic_audit.json).

Same sampling seeds do not imply matched rendered inputs or native inference-delivery timing.
Command bounds and measured motion are reported separately; limiter does not guarantee measured stopping distance.
Physical outcomes come only from frozen strict free-body/support metrics, never completion flags.
This helper does not identify colliding actors or certify physical sensor calibration.
