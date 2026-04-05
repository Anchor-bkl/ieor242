#!/usr/bin/env bash
#SBATCH -A mth250011p
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -t 01:00:00
#SBATCH -J hw3_smoke
#SBATCH -o slurm-smoke-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Small run to verify end-to-end pipeline quickly.
bash run.sh \
  --max_steps 300 \
  --eval_interval 100 \
  --save_interval 150 \
  --eval_batches 10 \
  --max_train_stories 200000 \
  --max_valid_stories 5000
