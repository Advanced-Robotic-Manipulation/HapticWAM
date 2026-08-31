#!/bin/bash
TASK=${1:?task}
~/phantom-icra-2027/phantom/.venv/bin/python ~/phantom-icra-2027/snap.py "$TASK"
export DISPLAY=:1
pkill -f "eog" 2>/dev/null
setsid nohup eog -f /tmp/snap.png >/dev/null 2>&1 < /dev/null &
echo "on screen: training $TASK vs now"
