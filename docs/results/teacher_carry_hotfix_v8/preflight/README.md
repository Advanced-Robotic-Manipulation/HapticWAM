# Frozen source and CPU evidence

`cpu_assembled_tests.json` and its exact 180-byte log record **100 passed** against the assembled runtime using CPU mocks. `cpu_suite/` preserves the actual test files and fixture bytes listed by that record. Tests were not rerun during artifact recovery. `cpu_frozen_plan.json` preserves the reviewed six-case launch plan; draft preflight is retained as preparation history, not as the final experiment contract.

`runtime_manifest.json`, `driver_manifest.json` and `inference_manifest.json` are the original frozen inventories. `runtime_delta/` contains each final runtime file listed in the runtime manifest delta. The unchanged V7 base archive and full V8 runtime remain on compute3. The exact source assembly is therefore distinct from the latest git branch; report and reproduce from these inventories.

`inference_paths_preflight.json` records the import/configuration-path check. `preserved_successes.json` independently compares earlier successful videos and mapping metadata with committed prior-study hashes. Checkpoint weights, full recordings, raw simulator streams and complete source trees are referenced by absolute compute3 paths and hashes; they are intentionally not copied into this compact report.
