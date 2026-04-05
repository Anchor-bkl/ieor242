#!/usr/bin/env bash
#SBATCH -A mth250011p
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -t 04:00:00
#SBATCH -J hw3_main_gpu
#SBATCH -o slurm-main-gpu-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Run the main model using GPU baseline defaults from run.sh,
# writing outputs to a dedicated folder for final report.
bash run.sh --out_dir out_main_gpu_baseline
