#!/usr/bin/env bash
#SBATCH -A mth250011p
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -t 04:00:00
#SBATCH -J hw3_tinystories
#SBATCH -o slurm-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
bash run.sh
