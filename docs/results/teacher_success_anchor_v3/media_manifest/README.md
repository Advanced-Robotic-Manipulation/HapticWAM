# Component review videos

All 12 component trials and both requested comparisons are rendered and verified. Full H.264 decode frame counts, MP4/metadata SHA-256 and byte counts, frozen runtime metric/hardware hashes, raw score-input hashes, and physical/tactile force-array fingerprints are recorded in the [manifest](component_video_manifest.json). The [final audit](final_media_audit.json) confirms all frozen scores agree exactly with displayed metrics and all sampled scene/state/tactile timestamps are causal.

Tactile panels display the saved native runtime gel pixels, explicitly labelled simulated and uncalibrated. Force graphs show physical packet normal-contact load; the tactile panels separately label the gel-proxy normal force. Comparisons use the same 60 s simulation clock; an ended run retains an explicit `STOP ... | FRAME FROZEN at ...` label. Each two-panel comparison has 901 frames at 15 Hz and resolution 1920×690. Individual reviews use their recorded observation duration.

| Review | Frames | MP4 bytes |
|---|---:|---:|
| baseline/teacher__fixed_anchor__seed904301 | 252 | 999,413 |
| baseline/teacher__fixed_anchor__seed904302 | 266 | 1,066,627 |
| delivery_feedback_only/teacher__fixed_anchor__seed904301 | 528 | 2,044,999 |
| delivery_feedback_only/teacher__fixed_anchor__seed904302 | 249 | 853,417 |
| finish_only/teacher__fixed_anchor__seed904301 | 271 | 1,057,173 |
| finish_only/teacher__fixed_anchor__seed904302 | 266 | 1,007,020 |
| gel_v2_only/teacher__fixed_anchor__seed904301 | 843 | 2,964,526 |
| gel_v2_only/teacher__fixed_anchor__seed904302 | 449 | 1,668,742 |
| live_veto_only/teacher__fixed_anchor__seed904301 | 260 | 996,151 |
| live_veto_only/teacher__fixed_anchor__seed904302 | 389 | 1,360,202 |
| wrist_distal_only/teacher__fixed_anchor__seed904301 | 255 | 1,020,860 |
| wrist_distal_only/teacher__fixed_anchor__seed904302 | 267 | 1,064,674 |
| baseline_vs_gel_v2_seed904301 | 901 | 4,428,405 |
| gel_v2_both_seeds | 901 | 4,997,476 |

Total: 14 MP4s, 25,529,685 bytes. 84 frozen score-input files verified across 12 distinct trials. Bulk MP4s, per-frame mapping JSONs and first-frame PNGs remain on compute3; no large recordings were copied locally.

Remote directory:

```text
/home/physicalai/phantom-icra-2027/sim/waffles/runs/teacher_success_anchor_v3/video_reviews
```

The two selected comparison files are:

```text
comparisons/baseline_vs_gel_v2_seed904301.mp4
comparisons/gel_v2_both_seeds.mp4
```

Download either file and its mapping JSON when needed; preserve the verified bytes:

```bash
rsync -a compute3:/home/physicalai/phantom-icra-2027/sim/waffles/runs/teacher_success_anchor_v3/video_reviews/comparisons/baseline_vs_gel_v2_seed904301.mp4 ./
rsync -a compute3:/home/physicalai/phantom-icra-2027/sim/waffles/runs/teacher_success_anchor_v3/video_reviews/comparisons/baseline_vs_gel_v2_seed904301.json ./
rsync -a compute3:/home/physicalai/phantom-icra-2027/sim/waffles/runs/teacher_success_anchor_v3/video_reviews/comparisons/gel_v2_both_seeds.mp4 ./
rsync -a compute3:/home/physicalai/phantom-icra-2027/sim/waffles/runs/teacher_success_anchor_v3/video_reviews/comparisons/gel_v2_both_seeds.json ./
```

The [render wrapper](render_component_reviews.py) called only the frozen presentation helper with cv2/libx264, sequentially at nice 19 with CPU affinity 30–31 and thread limits. It started PID 2257673 and finished normally. No simulator, model, GPU, hardware, frozen source or raw trial mutation occurred. The [final audit helper](audit_media.py) checks bytes and mappings without redundantly decoding already verified videos. Presentation helper SHA-256: `e4e51ecb40ac12ff4dc2664c65a33a9a5a1560d5080bf44fb1a22fef95bef054`.
