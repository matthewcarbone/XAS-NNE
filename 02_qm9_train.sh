#!/bin/bash -l

PID_FILE=saved_pid.txt
if [ -f "$PID_FILE" ]; then
    rm "$PID_FILE$"
fi

# Random O split
python3 02_qm9_train.py --data-path data/qm9/ml_ready/XANES-220710-ACSF-O-RANDOM-SPLITS.pkl > Logs/XANES-220710-ACSF-O-RANDOM-SPLITS.log
echo $! >> "$PID_FILE"
wait

# Random N split
nohup python3 02_qm9_train.py --data-path data/qm9/ml_ready/XANES-220710-ACSF-N-RANDOM-SPLITS.pkl > Logs/XANES-220710-ACSF-N-RANDOM-SPLITS.log &
echo $! >> "$PID_FILE"
wait

# Random C split
nohup python3 02_qm9_train.py --data-path data/qm9/ml_ready/XANES-220710-ACSF-C-RANDOM-SPLITS.pkl > Logs/XANES-220710-ACSF-C-RANDOM-SPLITS.log &
echo $! >> "$PID_FILE"
wait