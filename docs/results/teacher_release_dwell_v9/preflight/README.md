# Preparation evidence

`cpu_two_block_plan.json` records a successful CPU-only construction of exactly four trials using 5,104 verified pinned files. `ready_for_root_review.json` records the two campaign hashes, launcher/supervisor hashes and unchanged V8 source manifests. `resource_preflight.json` records the dedicated free port, output absence, GPU occupancy and disk space before execution. No model, Isaac instance or hardware was launched during these checks.

`abba_rejected.json` records the earlier single-seed campaign validation failure. This did not create a policy trial and does not add to the four-trial denominator. The original rejected ABBA configuration files are retained under `../launchplan/` solely as preparation history; `launch_dwell.py` and the final binding select only the two two-seed campaigns.

`cpu_final_plan.json` records the authorized final binding and 5,106 verified pins immediately before the single launch. The two added pins relative to the first successful preflight were that preflight plan and resource-check record; the runtime, driver and inference sources remained unchanged. Final post-run pin and process checks are in `../completion/completion_audit.json`.
