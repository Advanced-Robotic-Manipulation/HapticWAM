# Independent check of recorded-command mechanics

**All21 shared trace arrays are bitwise equal at all501 common simulation timestamps** between the original successful v1 case and `teacher_success_anchor_v3/accepted_command_replay`. This includes q/qd, TCP, measured gripper, packet pose/orientation, pad forces/contact geometry and independent bin/robot support. Effective scene configuration and complete initialization JSON also match.

Both files contain502 rows, but their final samples differ in time: original33.376 s, replay33.372 s. Those two rows were excluded from the common-time comparison. Their tiny positional/orientation differences therefore do not demonstrate mechanical nondeterminism. A future requested duration33.38 s can include the original endpoint; no rerun was needed for this audit.

An independent integer-nanosecond causal lookup checked every one of8344 physics ticks against the original3923 submitted-command rows. The replay reader selects the exact recorded q and finger targets on8343 ticks with available commands; at t=0 no command has yet been submitted. After the last command at31.380 s it holds the final targets. Logged scene-rate target q matches the source targets exactly after the logged dtype cast. No future command or interpolation is needed.

The frozen scorer independently reproduces all original object stages and event times, including acquisition8.000 s, lift12.668 s, carry15.336 s, release24.400 s and **support-verified full placement25.200 s**. The support filters are the same13 robot rigid bodies and five bin bodies. Over25.2–27.2 s, packet–robot contact is identically0 N; packet–bin support averages0.343097 N, compared with packet weight0.343350 N.

This establishes a reproducible simulated mechanical response to the original recorded drive commands. It is **not a fresh policy success**: the reader makes no new inference or safety decisions, and the scorer correctly reports `valid_for_scoring=false` with only `run_mode_is_not_policy`. The fresh closed-loop repeats' RGB and timing differences remain relevant.

[Numeric audit and hashes](mechanics_audit.json) and the [read-only CPU helper](audit_mechanics.py) preserve the calculation. No simulator, model or hardware was launched, and no frozen file or threshold was changed.
