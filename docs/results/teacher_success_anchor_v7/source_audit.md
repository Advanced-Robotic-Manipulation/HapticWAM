# Isolated RPC delivery source audit

Prepared an independent copy of the immutable limiter-v6 source. No model, GPU, Isaac, hardware, or campaign was launched for this audit.

The only changed files are `phantom/sim/policy_adapter.py` and `tools/sim/run_waffles.py`. All 4351 original non-cache files were inventoried; the original source was rehashed unchanged after copying. The shared limiter helper and every other source/asset file are unchanged. The source retains minimal gel-v2, FINISH, historical request-time veto, pad-only wrist proxy, and the existing default-off limiter.

`--policy-delivery-clock rpc_wall` measures the complete synchronous `policy.replan` call. It excludes observation callbacks and runner bookkeeping, preserves the native action grid and Plan latency/CPK token, and delays only delivery. Existing submit logic skips/rebases expired head samples. The default remains `native`; no latency override is allowed with `rpc_wall`.

Both complete source files pass the default-native projected-AST comparison. The actual frozen v6 and new v7 adapters also produced identical deterministic CPU traces for native delivery with an explicit latency override. Additional actual-v7 CPU tests passed late activation/rebasing, CPK token and native grid preservation into the next request, proposal immutability, expired-chunk rejection, cancellation on stop, four invalid clocks, override rejection before inference, CLI forwarding, and infrastructure-invalid classification. See [CPU audit](cpu_delivery_audit.json). These are stub-policy tests; numerical Isaac trajectory parity and network/model behavior have not been tested here.

The default path adds only explicit delivery-clock metadata. It does not silently enable RPC delivery or the limiter. Changing to full-client delivery may change physical outcomes and must be evaluated as a new declared condition.

- Source: `/home/physicalai/phantom-icra-2027/sim/waffles/source_teacher_anchor_rpc_v7`
- [Source manifest](source_manifest.json), SHA256 `1165cb34c53444c89ea76a78c9687e508adfd70b22570c2278060e83afd37084`
- Canonical file digest: `4a37b42124a6e2eff45a0900ef363f18838e0d15362328c2ae84f6f76d1b5897`
- Adapter SHA256: `cc5d424fa561ebeef06b8d0704acddabf7bc46819954f1913b9254fab238fd1e`
- Runner SHA256: `def8a003ec7c00e32254612c1ae18c0b94b6e2553f277e5e6b12b24a0ce4bd8f`
- Limiter helper SHA256: `41c0c6beebd805dda4d9bf7c66e507491eb1121cdb5cf70a701ce845b37e027f`
- Reviewed donor commit: `5c2f76b`; [adapter delta](policy_adapter.patch), [runner delta](run_waffles.patch).

Reproduce the CPU audit on compute3 with its existing PHANTOM Python:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/physicalai/phantom-icra-2027/phantom/.venv/bin/python /home/physicalai/phantom-icra-2027/sim/waffles/runs/teacher_success_anchor_v7/source_inputs/audit_rpc_cpu.py --source /home/physicalai/phantom-icra-2027/sim/waffles/source_teacher_anchor_rpc_v7 --baseline-source /home/physicalai/phantom-icra-2027/sim/waffles/source_teacher_anchor_limiter_v6 --out /tmp/rpc_delivery_readonly_audit.json
```

The output path must not already exist. [Assembler](assemble_rpc_source.py) and [CPU audit helper](audit_rpc_cpu.py) are outside frozen runtime source; source creation refuses existing output and changes no campaign protocol.
