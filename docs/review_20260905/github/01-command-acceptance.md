The executor advances `_last_cmd` before `URArm.servo_l()` reports whether the proposed command was actually sent. The driver's default-on IK guard can silently hold; the optional limiter can shorten a target. This makes subsequent commands use an anchor the arm never received. A review probe also reproduced empty IK reaching `servoJ([])` with the limiter disabled.

Related: #4, especially its accepted-pose and infeasible-anchor items. This issue extracts a bounded prerequisite; it does not require enabling or fully redesigning the limiter.

Implementation should return an explicit accepted/held/limited/rejected result with the actual streamed target and reason. Maintain separate proposed, streamed and measured states, and update the executor from the actual driver result. Validate joint solutions and the servo return status before recording a successful command.

Acceptance criteria:

- A silent-IK-rejection fixture cannot advance the executor's streamed-command anchor.
- A shortened command uses its actual streamed pose as the subsequent anchor.
- Empty/nonfinite/invalid-sized IK never reaches servoJ; a failed servo submission cannot be recorded as successful.
- Repeated rejection has a bounded, explicit halt path and useful reason.
- Existing successful playback and mock-driver paths retain their intended semantics.
- Regression tests exercise the actual executor and real driver with injected RTDE responses, including consecutive replans; isolated fake-controller assertions alone are insufficient.

Use the existing limiter tests and recording infrastructure to build the small offline harness. Keep measured-state lag separate from command acceptance. Tests with synthetic IK establish interface behavior, not physical UR3 feasibility or an 8 ms real-controller timing guarantee.

Sources: [executor](https://github.com/Advanced-Robotic-Manipulation/phantom/blob/f64aea89e2ec48c6975a8806476c29170beac189/phantom/deploy/executor.py#L398), [driver](https://github.com/Advanced-Robotic-Manipulation/phantom/blob/f64aea89e2ec48c6975a8806476c29170beac189/phantom/drivers/real/ur.py#L428).
