The policy server services one connected client until disconnect. While that client remains connected, a second client's identity probe cannot complete. PICK times out the probe and can treat the busy server as absent, loading a local model and attempting another control session.

September 4 logs show an ftA connection from 19:40:52 to 19:47:19; a launch during that interval reports only a v5 server, loads ftA locally, and fails three RTDE ownership checks before creating an episode. The team also reports suspended deployment processes in #4. The logs do not independently identify the register owner. The historical uncaught accept/authentication EOF is already fixed; this issue concerns busy status and exclusive control ownership.

Acceptance criteria:

- Model identity and idle/busy status remain available independently of a long-lived inference client.
- The launcher distinguishes busy, unreachable and absent states. A busy or ambiguous probe cannot silently initiate a competing robot-owning fallback.
- Acquire an exclusive rig lease before dashboard/preflight/control calls. Define its scope across every supported launch path and control host; a process-local counter alone is insufficient.
- A second process reports the owner/lease state without sending any robot-control call.
- Normal shutdown and crash recovery have explicit lease semantics; do not silently kill processes or steal a still-valid lease.
- A regression uses the actual localhost protocol with two clients, plus two-process ownership checks with a fake control endpoint. It proves both information-query responsiveness and zero control calls by the losing process.

Related: #4's deployment hygiene item and the execution prerequisites in the review. No model inference or robot is required to reproduce the protocol/ownership behavior.

Sources: [server loop](https://github.com/Advanced-Robotic-Manipulation/phantom/blob/f64aea89e2ec48c6975a8806476c29170beac189/phantom/inference/remote.py#L225), [PICK fallback](https://github.com/Advanced-Robotic-Manipulation/phantom/blob/f64aea89e2ec48c6975a8806476c29170beac189/tools/rig/PICK.sh#L49). Retained logs: `compute3:~/phantom-icra-2027/logs/serve_ftA.log` and `tmp-terminal.txt`.
