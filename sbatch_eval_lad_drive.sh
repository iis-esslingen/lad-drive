#!/usr/bin/env bash
set -eu

REPO=$(pwd)
LOGDIR="$REPO/eval_logs"
mkdir -p "$LOGDIR"

JOBNAME="eval_tiny"

sbatch --job-name="$JOBNAME" \
       --partition=gpu1 \
       --gres=gpu:1 \
       --cpus-per-task=48 \
       --time=72:00:00 \
       --exclude=gpu104 \
       --output="$LOGDIR/%x_%j.out" \
       --error="$LOGDIR/%x_%j.err" \
       --wrap "$(cat <<'BASH'
echo "JOB ID $SLURM_JOB_ID"

source ~/.bashrc
conda activate lad_drive
cd "$REPO"

bash leaderboard/scripts/run_evaluation_lad_drive.sh benchmark_tiny

wait
echo "JOB FINISHED"
BASH
)"

JOBNAME="eval_short"

sbatch --job-name="$JOBNAME" \
      --partition=gpu1 \
      --gres=gpu:1 \
      --cpus-per-task=32 \
      --time=72:00:00 \
      --output="$LOGDIR/%x_%j.out" \
      --error="$LOGDIR/%x_%j.err" \
      --wrap "$(cat <<'BASH'
echo "JOB ID $SLURM_JOB_ID"

source ~/.bashrc
conda activate lad_drive
cd "$REPO"

bash leaderboard/scripts/run_evaluation_lad_drive.sh benchmark_short

wait
echo "JOB FINISHED"
BASH
)"

JOBNAME="eval_long"

sbatch --job-name="$JOBNAME" \
      --partition=gpu1 \
      --gres=gpu:1 \
      --cpus-per-task=48 \
      --time=72:00:00 \
      --output="$LOGDIR/%x_%j.out" \
      --error="$LOGDIR/%x_%j.err" \
      --wrap "$(cat <<'BASH'
echo "JOB ID $SLURM_JOB_ID"

source ~/.bashrc
conda activate lad_drive
cd "$REPO"

bash leaderboard/scripts/run_evaluation_lad_drive.sh benchmark_long

wait
echo "JOB FINISHED"
BASH
)"