# Demonstration and training evidence for ftA1500

The ten complete native waffles demonstrations show a substantially more compact carry/release path than the simulated teacher. **None crosses the 468 mm wrist-extension bound; their maximum is 428.84 mm**, including approach and retreat. This supports investigating learned orientation/path drift and controller reach handling. It does not support relaxing the measured safety bound. The available demonstrations contain full pick/place behavior: task text is `waffles`, and measured closure releases after lift/carry in all ten. Exact historical training membership remains unresolved, as detailed below. A script named `PICK.sh` does not establish lift-only training.

This CPU audit read recordings and source on compute3 without constructing a model or changing recordings, frozen studies, runtime, training or hardware. [Summary](summary.json), [per-episode CSV](demonstrations.csv), [native trajectory/tactile audit](demonstration_audit.json), and [window audit](training_window_audit.json) retain source/data hashes and exact native phase samples. Additional current loader/model source hashes are in [source_contracts.json](source_contracts.json). Phase labels below are tactile/gripper/TCP proxies, not independent object-pose ground truth. The ten episodes have success metadata; canonical August22 `5928` also has the previously reviewed full pick/place video.

## Recorded path versus current experiment

| Quantity | Ten complete August22 demonstrations | Current V5/V7 anchor |
|---|---|---|
| Actual source | Teleoperation recordings, green packet | September4 teacher deployment5016 initial robot state; beige packet in fitted post-hand position plus the declared −10/−10 mm offset |
| Initial TCP Z | 296.98–357.09 mm | 338.14 mm |
| Initial closure, 0=open | .2549–.4275 | .0784 |
| Start pose difference from Sept4 | 13.9–44.5 mm translation; 6.1–10.95° geodesic rotation | Reference |
| Entire-episode wrist-radius maximum | 404.90–428.84 mm | Kinematics audit reports simulated successful cases at 465.06/467.40 mm |
| Largest measured joint speed per episode | .61–.82 rad/s | See separate simulated trajectory audit |
| Carry peak TCP Z | 301.73–358.09 mm | Simulated successful cases reach approximately 407/412 mm |
| Release TCP XYZ | X −377.79…−349.15, Y 67.26…100.25, Z 95.92…110.48 mm | See actual simulated release poses; do not substitute the demo pose as a command |
| Release rotation vector | X −2.089…−1.900, Y −1.230…−1.092, Z .517… .618 rad | Compare rotations geodesically, not just individual components |

The radius uses exactly the native safety formula `sqrt(a2²+a3²+2*a2*a3*cos(q_elbow)+d4²)` with UR3 constants .24365/.21325/.11235 m. There are 26,572 native joint samples across the ten recordings. At release their wrist radii are only 289.00–313.19 mm. The separate kinematics audit finds some simulated failures extending the wrist while **descending**, with orientation increasing extension more than translation reduces it. Therefore a vertical lift cap alone would not address every observed failure.

Canonical `5928` has bilateral |SDK Fz|>2.5 N for .15 s at 7.544 s, a sustained TCP rise of 30 mm at 7.984 s, and the measured closure-release proxy at 13.817 s. Its release TCP is `[-.349150,.071082,.105660,-2.074308,-1.092283,.540739]`; measured q and timestamps are retained in the JSON. Across the ten, corresponding release proxies occur at 13.817–16.394 s. These thresholds describe the recorded phase; they are not a new simulator score or claim that tactile threshold crossing exactly equals acquisition.

The August canonical packet fit is `[-.366715,-.259618,.07]`, yaw .44229 rad, green. The V5/V7 anchor center is `[-.393707,-.293069,.07]`, yaw .316566 rad, beige: approximately 43 mm farther in the table plane. The September source also includes a real human packet adjustment at 4.4–5.6 s, omitted by the static post-hand reconstruction. The sessions are explicitly different fixtures. Neither the geometry nor an inferred sensor match should be silently transferred between them. See the [prior camera/start audit](../../teacher_v2_design/README.md).

## Training contract and supported limitations

The current ftA1500 bytes match the previously CPU-inspected payload: SHA-256 `67c93287123e447b85bf32385660f1a99d6a8b0ad5e995727ac92a680543893e`, step1500, teacher architecture, finite EMA. The saved normalizers match canonical hash `42153b79ceb6a6c711296323c6b171e42929eb5dfe4018198d63d2cd3ace8c48`. Saved training uses batch4/accumulation2, EMA .995, 30% pre-close window bias, photometric augmentation1, commit-band multiplier1, no tactile-SSL initialization, and saved NFE5. Deployed NFE1/K4 is an explicit inference recipe, not a saved training guarantee. The archived logs previously reviewed reach step3000 without reported NaNs; the old batch-one weight-cancellation and collapsed-event-label diagnoses do not apply to this inspected configuration.

The source pins in the JSON refer to **current live HEAD `0259ed3355f2ad3078778e7cc347b100775e06d3`**, not a recovered exact training checkout. Current recording/loader/executor semantics are consistent: six base-frame TCP increments (meters plus continuity-guarded rotation-vector component increments), followed by **absolute** gripper closure. The executor sums the pose increments and interpolates at 10 Hz. They are not velocity commands to multiply by dt again, absolute poses, or relative gripper changes. Native demo action intervals have median 100.26–100.41 ms; p95 is 104.64–106.25 ms. Nearest action-time measured-TCP differencing has median per-axis translation RMSE .184/.551/.514 mm; independent recording timestamps prevent exact reconstruction by nearest matching, but do not show a factor-of-ten action-unit error.

The saved normalization is `(value−mean)/std`; inference reverses it before execution. Action means are near zero in the six motion channels and .42186 for closure. Pad-wrench baseline rows resolve to zero for this checkpoint. Wrist and fields are normalized once; `contact_state` is raw apart from that disabled pad baseline. Safety rolling wrist tare is a separate operation. Tiling the current RGB across inference video slots is intentional: only the first latent frame is conditioning. It is not, by itself, a training/inference image-history bug. The [existing preprocessing audit](../../teacher_v2_design/preprocessing_audit.md) and saved-input equality tests already constrain this class of explanation.

Two narrower training limitations remain supported:

- **Startup sampling:** the current loader requires 1.6 s of past context and 3.125 s of future stream overlap. On these recordings the earliest valid anchors are 1.642–1.720 s after the first joint sample. By then the arm has already descended roughly19–107 mm in the retained 10 Hz trajectory. First-request zero-motion/padded history at an actual recording start is therefore not directly represented by the first admissible window of these ten examples. This does not prove the pose never appears elsewhere in training. A deployment-matched startup-window augmentation is a testable remedy.
- **Training objective:** the recorded recipe preferentially samples `[first_close−1.5, first_close−.2]` for30% of windows; the rest is uniform. Release remains in the admissible range, but has no analogous explicit sampling bias. The inspected action objective is denoising velocity MSE on 16 future steps, not a joint-limit, collision or whole-task placement objective. This supports measuring phase coverage and adding reach/path diagnostics before claiming that more training steps will solve the problem. It does not establish that release was absent or that a particular loss change is already effective.

Historical training membership is unresolved. The retained root contains 250 success-labelled demos, but **240 are incomplete inspection mirrors**: their later Zarr chunks and all timestamp arrays are absent, so reading the declared shape returns fill zeros. Only their stored first rows support the old start inventory; their later trajectory values are excluded here. These are **not proved to be the paths/bytes consumed by the historical optimizer**. The expanded training manifest is absent at this root, and the inspected payload saves `split=train` but no exact episode list/root or training source revision. Do not diagnose “training on zeros” from this mirror. The original and expanded-validation manifest hashes are retained; the ten complete examples are listed in expanded validation, without proving absence from every historical pool.

## Tactile phase comparison

Values below are the range of **per-episode medians** across the ten native recordings. Phases use native timestamps; fields use the same derived depth mask and slip implementation as the loader. SDK resultant forces use the recorded driver convention, while distributed-force channels remain uncalibrated.

| Channel | Initial unloaded interval, left/right | Lift/carry, left/right |
|---|---|---|
| SDK contact area, mm² | 0 / 0 | 0–40.89 / 0–19.90 |
| Depth-mask fraction, absolute depth>.05 mm | .0010–.0102 / 0 | .151–.345 / .0847–.262 |
| Absolute peak depth per field, mm | .0565–.1186 / .0161–.0283 | .1987–.4939 / .1913–.3486 |
| Raw recorded Fz, N | −1.718…−.966 / −.0529…−.0133 | −25.51…−8.14 / −14.30…−5.29 |
| Derived slip statistic | .0302–.0751 / 0 | 0–.00119 / 0–.02174 |

The full ranges and six-component wrench statistics are in [summary.json](summary.json). Lateral wrench components are often nonzero during real carry. Left/right no-contact residuals differ substantially. **SDK area can be zero while a large depth-mask fraction is present in actual loaded data** (e.g.5963-left and6128-right). Thus `mask_fraction ×972 mm²` is not an established identity for SDK contact area; discrepancies cannot alone prove a proxy bug. Match the joint distribution of area, depth, wrench, slip and image shape against measured states. A normal-force-only synthetic Gaussian and fixed baseline do not establish that joint distribution or physical shear/torque transfer.

## Concrete next tests

1. Compare the native and simulated carry paths by object-relative TCP position, SO(3) orientation and wrist radius at matched contact/lift phases. First replay measured demo commands through the same controller/scene to verify that the compact real path is retained. Then test a reach-aware proposal/servo guard against the same saved failures, leaving the actual stop bound unchanged. CPU IK alone cannot certify collision-free physical execution.
2. Add an explicitly separate, deployment-matched startup-window dataset using the native builder's history-padding rules and actual measured initial closure. Check input/target equality on recorded episodes before retraining; keep the current scored studies immutable. Compare fixed scene and same starts/seeds so a session change cannot masquerade as an improvement.
3. Calibrate tactile channels jointly using unloaded and measured force ramps on both pads, and held contacts through the real carry orientation range. Fit against measured force/area/depth/shear behavior, not against a latch or success threshold. Test the resulting mapper first on saved contacts, then on separately declared fresh policy trials.
4. Recover and freeze the exact training episode manifest and code/config provenance, report sampled approach/lift/carry/release coverage, and evaluate closed-loop placement separately from offline endpoint MSE. If adding release sampling or constrained imitation, preserve a new held-out session and the current failed approaches as regression tests.

## Reproduce

Run the two read-only helpers from the worktree; compute3's live Python supplies NumPy/Zarr/PyTorch. Redirect to fresh files when preserving this audit. They do not initialize a model or GPU.

```sh
ssh compute3 'cd /home/physicalai/phantom-icra-2027/phantom && OPENBLAS_NUM_THREADS=1 .venv/bin/python -' \
  < docs/results/teacher_behavior_debug_20260907/data/audit_demonstrations.py > /tmp/demonstration_audit_fresh.json
ssh compute3 'cd /home/physicalai/phantom-icra-2027/phantom && OPENBLAS_NUM_THREADS=1 .venv/bin/python -' \
  < docs/results/teacher_behavior_debug_20260907/data/audit_training_windows.py > /tmp/training_window_audit_fresh.json
```

`summary.json` and the CSV are produced locally by `summarize_audit.py` using the two preserved JSON files and linked checkpoint/start evidence. The local interpreter requires NumPy and SciPy. No hardware launch scripts or simulator runs are part of this audit.
