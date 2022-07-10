#!/bin/bash -l

WORKDIR=/hpcgpfs01/work/cfn/mcarbone/XAS-NNE
# sbatch 02_qm9_train.sbatch.sh --data-path "$WORKDIR"/XANES-220710-ACSF-O-RANDOM-SPLITS-PCA-decomp-maxcol-21.pkl --print-every-epoch 50 -n 30 --n-gpu 2
# sbatch 02_qm9_train.sbatch.sh --data-path "$WORKDIR"/XANES-220710-ACSF-N-RANDOM-SPLITS-PCA-decomp-maxcol-25.pkl --print-every-epoch 50 -n 30
sbatch 02_qm9_train.sbatch.sh --data-path "$WORKDIR"/XANES-220710-ACSF-C-RANDOM-SPLITS-PCA-decomp-maxcol-23.pkl --print-every-epoch 50 -n 30 --n-gpu 1
