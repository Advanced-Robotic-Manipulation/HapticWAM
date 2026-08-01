# NUC changes 2026-08-01 (Claude, at Mikhail's request) — SYNC THESE TO THE REPO

## 1. Session-start segfault FIXED — drivers/real/ur.py (backup: _backup_20260801_190800_segvfix/)
Kernel log showed 3 identical segfaults today (17:51, 18:18, 18:52) in rtde_control.so's
asio thread, each at SESSION START: the first collect process died constructing
RTDEControlInterface against a robot whose old control script was still dying ("1st start
after a fault fails, 2nd works" was never a polite failure). Changes:
- _preflight_stop_old_script(): dashboard 'stop' + wait for receive getRuntimeState()!=PLAYING
  + settle, BEFORE constructing. Best-effort, falls through to the retry loop.
- _probe_construct(): first construction happens in a throwaway subprocess — a residual
  constructor segfault kills only the probe (rc=-11 logged), never the recorder.
- program_running(): now answered from the RECEIVE side (getRuntimeState()==2 PLAYING).
  The old body probed ctrl.isProgramRunning() — the exact call your own 2026-07-27 comment
  says segfaults on a dead-script interface. Landmine removed (it had no callers, now safe to call).
- Verified live: module imports, dashboard_client present, runtime_state read off the real
  robot (=1 STOPPED idle, mode 7). Cost: ~1-2 s extra at session start. No API changes.

## 2. Rerun viewer memory cap — use ./viewer.sh
The bare `rerun` viewer accumulates ALL frames: it hit 24 GB RSS today and swapped the box
(the camera lag). `~/phantom-icra-2027/viewer.sh` launches it with --memory-limit 4GB.
The currently running viewer is already the capped one. Server side was always capped (512MB).

## 3. recover_episodes.py (new util)
Finalizes episodes stranded with meta status "recording" after a crash (data is append-only
zarr — nothing is ever actually lost). Ran today: 0 stranded (all 7 Carton eps in
20260801_184043 were already finalized, success=true).

## 4. HF auto-upload (new)
Private dataset repo: hf.co/datasets/armteam/phantom-episodes.
~/phantom-icra-2027/hf_upload_episodes.py + systemd user timer phantom-hf-upload.timer
(hourly). DEFERS whenever a collection process is running; skips sessions modified <15 min
ago; manifest ~/.phantom_hf_uploaded.json. Uploads collect/ + episodes/ + the external
drive's phantom_episodes/. Token: ~/.cache/huggingface/token (businesslion, armteam org).

## Note
This tree has diverged from origin/main (collect script, data_collect/, these fixes).
Please sync back to github.com/Advanced-Robotic-Manipulation/phantom when convenient —
at minimum the ur.py fixes belong upstream.
