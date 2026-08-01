#!/bin/bash
# PHANTOM rerun viewer with a hard memory cap (the uncapped viewer reached
# 24 GB on 2026-08-01 and swapped the box -> camera lag). Use THIS, not bare rerun.
exec ~/phantom-icra-2027/.venv/lib/python3.10/site-packages/rerun_sdk/rerun_cli/rerun --memory-limit 4GB "rerun+http://127.0.0.1:9878/proxy" "$@"
