#!/bin/bash
# after an e-stop / power cycle: deactivate -> activate (calibration stroke, keep fingers clear) -> open
cd ~/phantom-icra-2027/phantom && exec .venv/bin/python -m phantom.scripts.gripper_ctl reset --hardware configs/hardware.nuc.yaml "$@"
