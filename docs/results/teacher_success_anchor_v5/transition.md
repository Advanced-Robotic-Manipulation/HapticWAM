# Screen-to-confirmation transition

[The one-shot wrapper](transition_to_confirmation.py) is separate from the frozen simulator, driver and review helper. It was prepared while the screen was active; it does not change the design or select from partial outcomes. Eight focused CPU guard tests pass. No wrapper execution or GPU launch was performed during preparation.

The screen identity is pinned to PID2306707, Linux start tick102173374 and boot`9b1fae31-5718-4674-bf3b-f3b413a89cc8`, with the exact argv in the hash-pinned `screen_launcher.json`. A reused PID, changed argv or reboot aborts. The wrapper polls every30s, requires the controller to exit and all24 original attempts to be completed, exit0 and valid under the frozen thresholds. It also checks that archived owned server processes are absent and port7799 has no listener. It never kills a process.

Before confirmation it verifies all120 admitted runtime/input bindings, all343 frozen driver files, the five review modules, checkpoint bytes and the screen's recorded source/core/input hashes. Selection must be complete and hash-linked. The unchanged review helper derives the two selected policies and the twelve reserved seeds904501–904512. A separate plan-only invocation checks all24 confirmation cells before the executed invocation. Existing selection or confirmation outputs cause an abort; no attempt is resumed, replaced or retried.

Authoritative remote paths:

```text
/home/physicalai/phantom-icra-2027/sim/waffles/runs/teacher_success_anchor_v5/transition_to_confirmation.py
/home/physicalai/phantom-icra-2027/sim/waffles/runs/teacher_success_anchor_v5/transition/ledger.json
/home/physicalai/phantom-icra-2027/sim/waffles/runs/teacher_success_anchor_v5/transition/final_status.json
```

Root can first invoke the default CPU-only check, then launch the waiter once:

```bash
ANCHOR_V5_PY=/home/physicalai/phantom-icra-2027/phantom/.venv/bin/python
ANCHOR_V5_ROOT=/home/physicalai/phantom-icra-2027/sim/waffles/runs/teacher_success_anchor_v5

PYTHONDONTWRITEBYTECODE=1 "$ANCHOR_V5_PY" "$ANCHOR_V5_ROOT/transition_to_confirmation.py"
PYTHONDONTWRITEBYTECODE=1 "$ANCHOR_V5_PY" "$ANCHOR_V5_ROOT/transition_to_confirmation.py" --execute
```

Each child command has a separate log and recorded PID/exit code under `transition/`. The six-hour overall deadline or an interrupt records an abort and any still-running owned child PID; it deliberately does not kill that child or authorize an automatic retry. Root must inspect that ledger before any manual recovery.

Final confirmation scoring uses the frozen paired uncertainty and winner gates. A null winner is retained. The wrapper does not launch the arm extension, native hardware or any extra experiment.

Root launched this wrapper at 09:42:15 UTC as PID2367044. After all 24 screen cases passed the declared checks, it started confirmation at 10:05:54 UTC as child PID2415593. The invocation above is retained for provenance; do not start a second waiter for these existing outputs.
