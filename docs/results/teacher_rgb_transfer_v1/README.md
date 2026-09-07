# Preserved first RGB diagnostic attempt

The v1 attempt ended before any of the six diagnostic first-plan calls. The client treated the `busy=True` response to its own successful `configure({})` request as a foreign owner, then disconnected. Native protocol ownership begins with that configuration request; `info` alone is read-only. This was a client ownership-check error, not a model outcome. The dedicated server performed its synthetic warmup before this failure.

The [failure audit](audit.json), [client traceback](orchestration/client.log), [orchestration status](orchestration/status.json), [server exit metadata](server/server.json), and original preparation/ready metadata are preserved with hashes. There are zero saved seed proposal files and no `results.json`. Owned server PID 1547516 had exited when inspected.

The explicit [v2 retry](../teacher_rgb_transfer_v2/README.md) uses a bounded read-only preconnection `info` probe, then accepts native ownership acknowledgment for its own connection. A foreign owner and a competitor winning the subsequent race are still refused. Six-call ordering, seeds, teacher EMA, inference recipe and RGB/non-RGB inputs are unchanged. Four focused ownership/timeout regressions and the existing RGB/remote tests pass (16 total). No frozen primary study or score was changed.
