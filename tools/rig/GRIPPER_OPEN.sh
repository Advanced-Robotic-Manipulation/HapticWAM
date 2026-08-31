#!/bin/bash
# open the Robotiq between runs (activates first if the gripper lost activation)
cd ~/phantom-icra-2027/phantom && exec .venv/bin/python -m phantom.scripts.gripper_ctl open --hardware configs/hardware.nuc.yaml "$@"
