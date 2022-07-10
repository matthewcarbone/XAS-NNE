#!/bin/bash -l

# PID_FILE=saved_pid.txt
# if [ -f "$PID_FILE" ]; then
#     rm "$PID_FILE$"
# fi

python3 02_train.py --data-path data/qm9/ml_ready/XANES-220710-ACSF-O-RANDOM-SPLITS.pkl
# echo $! >> "$PID_FILE"
# wait
