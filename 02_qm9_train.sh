#!/bin/bash -l


# Passed via command line
RUN_WHAT="$1"
EXECUTABLE="${2:-echo}"

N_ENSEMBLES=30
PRINT_EVERY_EPOCH=50
N_GPU=1
MAX_EPOCHS=2000


run_random_qm9()
{
    WORKDIR=data/qm9/ml_ready/random_splits
    PCA="$1"
    if [ "$PCA" = true ]; then
        declare -a paths=(
            "XANES-220712-ACSF-O-RANDOM-SPLITS-PCA-decomp-maxcol-21"
            "XANES-220712-ACSF-N-RANDOM-SPLITS-PCA-decomp-maxcol-25"
            "XANES-220712-ACSF-C-RANDOM-SPLITS-PCA-decomp-maxcol-23"
        )
    else
        declare -a paths=(
            "XANES-220712-ACSF-O-RANDOM-SPLITS"
            "XANES-220712-ACSF-N-RANDOM-SPLITS"
            "XANES-220712-ACSF-C-RANDOM-SPLITS"
        )
    fi

    declare -a downsample_values=(0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1)

    for path in "${paths[@]}"; do
        for downsample_prop in "${downsample_values[@]}"; do
            "$EXECUTABLE" .02_qm9_train.sbatch.sh \
                --data-path "$WORKDIR"/"$path".pkl \
                --ensemble-name Ensembles/"$path"/"$downsample_prop" \
                --downsample-prop "$downsample_prop" \
                --print-every-epoch "$PRINT_EVERY_EPOCH" \
                --n-gpu "$N_GPU" \
                -n "$N_ENSEMBLES" \
                --max-epochs "$MAX_EPOCHS"
        done
    done
}

run_generalization()
{
    WORKDIR=data/qm9/ml_ready/by_total_atoms
    declare -a paths=(
        XANES-220711-ACSF-C-TRAIN-ATMOST-5-TOTAL-ATOMS.pkl
        XANES-220711-ACSF-N-TRAIN-ATMOST-7-TOTAL-ATOMS.pkl
        XANES-220711-ACSF-C-TRAIN-ATMOST-6-TOTAL-ATOMS.pkl
        XANES-220711-ACSF-N-TRAIN-ATMOST-8-TOTAL-ATOMS.pkl
        XANES-220711-ACSF-C-TRAIN-ATMOST-7-TOTAL-ATOMS.pkl
        XANES-220711-ACSF-O-TRAIN-ATMOST-5-TOTAL-ATOMS.pkl
        XANES-220711-ACSF-C-TRAIN-ATMOST-8-TOTAL-ATOMS.pkl
        XANES-220711-ACSF-O-TRAIN-ATMOST-6-TOTAL-ATOMS.pkl
        XANES-220711-ACSF-N-TRAIN-ATMOST-5-TOTAL-ATOMS.pkl
        XANES-220711-ACSF-O-TRAIN-ATMOST-7-TOTAL-ATOMS.pkl
        XANES-220711-ACSF-N-TRAIN-ATMOST-6-TOTAL-ATOMS.pkl
        XANES-220711-ACSF-O-TRAIN-ATMOST-8-TOTAL-ATOMS.pkl
    )

    for path in "${paths[@]}"; do
        "$EXECUTABLE" .02_qm9_train.sbatch.sh \
            --data-path "$WORKDIR"/"$path".pkl \
            --ensemble-name Ensembles/"$path"/0.9 \
            --downsample-prop 0.9 \
            --print-every-epoch "$PRINT_EVERY_EPOCH" \
            --n-gpu "$N_GPU" \
            -n "$N_ENSEMBLES" \
            --max-epochs "$MAX_EPOCHS"
    done
}



if [ "$RUN_WHAT" = "qm9-pca" ]; then
    run_random_qm9 true
elif [ "$RUN_WHAT" = "qm9" ]; then
    run_random_qm9 false
elif [ "$RUN_WHAT" = "gen" ]; then
    run_generalization
else
    echo "Unknown input"
fi
