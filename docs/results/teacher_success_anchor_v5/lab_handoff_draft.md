# Final teacher lab handoff — v5 confirmation

This path is retained for earlier links; the final operational documents are the
[one-page card](../../isaac_teacher_lab_card_20260907.md) and
[complete lab handoff](../../isaac_lab_handoff_20260907.md).

**Recommend ftA1500 teacher EMA, NFE1/K4, guidance 1 as the next attended lab
candidate. No validated reliable winner was established.** Reserved confirmation
only: ftA1500 acquired 12/12, lifted 7/12, carried 6/12, physically placed and cleanly
finished 1/12, and dropped 1/12; ftA3000 acquired 8/12, lifted/carried 4/12, placed 0/12
and dropped 0/12. No earlier screen, diagnostic or campaign counts are pooled.
[Full audit and uncertainty](confirmation/independent_audit.md).

The exact checkpoint, EMA/normalizer identity and inference settings are in the
[model table](model_settings_table.md). The native opt-in is
`--placement-controller-profile minimal_v5`, with `--terminal-veto` and an
explicit measured `--placement-release-config` with FINISH enabled.
[PR14](https://github.com/Advanced-Robotic-Manipulation/phantom/pull/14) contains
the port, introduced by commit 51457a36a8f3af3e4dbdc0100a8d6121aae70804.
[Controller CPU parity](native_controller_port.md) does not establish real
sensor/geometry/I/O qualification or installation on the live rig.

For the next bench session, measure the rig/pads/aperture/packet/box/camera and
unloaded/known-load sensors. Fix waffle placement, qualify one reviewed start
and release volume, then declare measured variable-arm-start attempts with fresh
seeds. Keep all attempts, physical object outcomes and controller stops distinct.
The simulator winner-only arm extension remains unexecuted; a proposed physical
calibration pilot is a separate study.

Rigid-packet compliance, peripheral contact, gel transfer and rolling wrist
baseline absorption are material [realism limitations](contact_force_audit.md).
The selected success includes a later historical-recovery contribution to final
aperture, disclosed in the [opening/FINISH audit](selected_success_audit.md).
V6 limiter and V7 full-RPC timing follow-ups are separate, not promoted settings.
No hardware run, live deployment update or safety-limit change was made here.
