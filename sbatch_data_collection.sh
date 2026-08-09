#!/usr/bin/env bash
set -eu

REPO=$(pwd)
LOGDIR=$REPO/data_collection/logs
mkdir -p "$LOGDIR"

for script in "$REPO"/data_collection/bashs/sub-3/todo_*; do
    name=$(basename "$script" .sh)

    echo "creating sbatch for $name"

    sbatch \
        --job-name="$name" \
        --partition=gpu1 \
        --gres=gpu:1 \
        --cpus-per-task=8 \
        --time=48:00:00 \
        --output="$LOGDIR/%x_%j.out" \
        --error="$LOGDIR/%x_%j.err" \
        --wrap "
            echo JOB ID \$SLURM_JOB_ID
            source ~/.bashrc
            . /home/es/es_es/es_fschmidt/miniconda3/etc/profile.d/conda.sh
            conda activate lmdrive
            cd \"$REPO\" 
            bash \"$script\"
            echo JOB FINISHED
        "
done
