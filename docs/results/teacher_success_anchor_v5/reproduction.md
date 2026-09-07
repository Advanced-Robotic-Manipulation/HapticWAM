# Reproducing the frozen teacher experiment

The executed simulator and external campaign driver have independent source
archives. These preserve the exact versions used by the experiment, including
native modules that differ from the later published simulator commit.

| Archive | Bytes | SHA256 |
|---|---:|---|
| `minimal_runtime.tar.gz` | 4,563,341 | `714ab9ea54398f12879ea93fd4d361f21cef97baf85568adb842eebdbfe568e3` |
| `experiment_driver.tar.gz` | 1,376,548 | `8c5f7e5f4650a8becb9407b51a9de64e9691790bd4ea6626cc670b76f27cae1c` |

Both archives are on compute3 under:

```text
/home/physicalai/phantom-icra-2027/sim/waffles/runs/teacher_success_anchor_v5/reproduction_archives/
```

The [archive manifest](reproduction_archives_manifest.json) lists every included
file and its hash. Runtime code, assets and configuration are included; model
weights, source recordings and the Isaac installation remain separate pinned
dependencies. The campaign's `runtime_contract` identifies the measured start,
September tactile baseline, prepared evidence, robot USD, hardware configuration
and native inference source hashes.

Retrieve and verify either archive before extraction into a new directory:

```bash
scp compute3:/home/physicalai/phantom-icra-2027/sim/waffles/runs/teacher_success_anchor_v5/reproduction_archives/minimal_runtime.tar.gz ./
sha256sum minimal_runtime.tar.gz
```

Use the [experiment commands](README.md) with a new output directory. Keep the
original experiment directories and source copies unchanged. The archived
configuration uses compute3 paths; relocating dependencies requires an explicit
new configuration and hash record. A repeated sampling seed alone does not
guarantee identical rendered observations, native latency or closed-loop motion.

CPU scoring uses the separately frozen review helper and
[review manifest](review_manifest.json). The simulator, campaign runner and
reviewer are deliberately recorded separately so a reporting update cannot
silently alter executed controller behavior.

The two gel-only development successes are available with synchronized saved
tactile inputs at:

```text
/home/physicalai/phantom-icra-2027/sim/waffles/runs/teacher_success_anchor_v3/video_reviews/comparisons/gel_v2_both_seeds.mp4
```

These videos establish those two physical placements. They do not enter the
prospective model comparison; later development repetitions also contain
retained failures.
