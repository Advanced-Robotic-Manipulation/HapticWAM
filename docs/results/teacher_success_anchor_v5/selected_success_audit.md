# Held-out successful illustration: ftA1500, seed 904510

This is a selected illustration from the complete 24-trial confirmation set.
It does not change the denominator, winner rule or result: **no clear simulator
winner**. The independently checked raw score and controller events are saved in
[selected_success_audit.json](selected_success_audit.json). The read-only CPU
[audit helper](selected_success_audit.py) pins the actual source and input hashes.

The frozen score records acquisition at 7.936 s, lift at 11.200 s, carry at
13.600 s, release in the bin at 23.600 s and strict supported placement at
24.268 s. There is no drop or actual safety stop. Peak per-pad packet normal
force is 6.384 N in the simulator; this is not a calibrated real-force claim.

The controller's separate FINISH event occurs at 25.076 s. Raw teacher plan 27
equals its delivered action array exactly. From 23.380 through 23.580 s, all
26 executor rows sample teacher closure commands between 0.43235 and 0.44894,
below the 0.45 opening threshold, with measured TCP inside the configured
release volume. The previously loaded latch remains engaged until the 0.2 s
sustained opening condition commits release at 23.580 s.

Measured unloaded/open dwell starts at 24.876 s and lasts 0.2 s. Its maximum
measured closure is 0.44780; six saved gel samples covering the recent window
have zero mapped normal force on both pads. Those gel loads remain uncalibrated
proxy values. At 25.004 s the historical terminal veto applies `recovery_open`
to plan 29, changing an already opening teacher proposal (0.319–0.344) to
0.232 and invalidating its CPK token. This occurs after physical placement was
confirmed. The final accepted-open gate therefore includes a native recovery
contribution; the final aperture is not attributed solely to the teacher.

The FINISH arm reference equals the measured TCP at 25.076 s and stays constant
for 4,366 execution rows through 59.996 s. There are no subsequent replans or
stops; maximum measured translational tracking error during hold is 2.76 mm.
The packet has zero robot normal-contact force throughout this final hold,
positive final bin support of 0.34355 N and less than 0.1 μm sampled displacement
after FINISH. Its final position is approximately
`[-0.422678, 0.084119, 0.072000]` m. These independent physical observations
support the strict score; FINISH itself does not inspect object state.

The copied review is 900 fully decoded frames at 15 Hz: a 60 s video with
actual telemetry through 59.996 s. Its saved tactile panels use exact runtime
pixels and causal sample mappings. Visual review of extracted frames at 8,
14, 24.333, 25.133, 30 and 59 s shows approach/acquisition, loaded transport,
packet resting in the box, and continued released hold. The status label first
shows `RELEASE COMPLETE / HOLD` after the actual 25.076 s controller event.

User-workspace output directory:

```text
artifacts/isaac_waffles/teacher_success_anchor_v5/video_reviews/confirmation/fta1500_nfe1_k4__successful_anchor__seed904510/
```

It contains `policy_review.mp4`, its unchanged mapping JSON, six PNG frames,
`selected_contact_sheet.jpg`, `local_media_audit.json` and a copy of the telemetry
audit. Downloaded video SHA-256:
`762017cc904af7cb49c1043b9e779666b1cccb92b3dbe932ec7720c3d002c9bf`.
Metadata SHA-256:
`c3c59126b9abf8207bbd9444f71aad62c73a8371befb19105e70e7e58ab866aa`.
