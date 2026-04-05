#!/usr/bin/env bash
set -euo pipefail

# Safety gate: avoid accidental training on a login node.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
	echo "ERROR: No SLURM allocation detected."
	echo "Use one of:"
	echo "  1) Interactive: salloc -A mth250011p -p GPU-shared --gpus=1 -N 1 -t 04:00"
	echo "  2) Batch: sbatch submit.slurm"
	exit 1
fi

host="$(hostname)"
if [[ "$host" == br* ]]; then
	echo "ERROR: Host appears to be a login node ($host)."
	echo "Please run on a compute node allocated by SLURM."
	exit 1
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$HOME/venvs/hw3}"

if [[ -d "$VENV_DIR" ]]; then
	# shellcheck disable=SC1090
	source "$VENV_DIR/bin/activate"
else
	echo "WARN: venv not found at $VENV_DIR; using current Python environment."
fi

mkdir -p "$PROJECT_DIR/out" "$PROJECT_DIR/data"

echo "Running on host: $host"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

if command -v nvidia-smi >/dev/null 2>&1; then
	echo "nvidia-smi -L:"
	nvidia-smi -L || true
fi

# If a GPU was requested from Slurm, require PyTorch CUDA to be available.
if [[ -n "${SLURM_GPUS:-}" || -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
	python - <<'PY'
import sys
import torch
print(f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda}, cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
		print("ERROR: Slurm allocated GPU(s), but PyTorch CUDA is unavailable.")
		print("Likely CUDA-driver/runtime mismatch. Reinstall a compatible PyTorch build (e.g., cu126).")
		sys.exit(2)
PY
fi

echo "Tip: set TRAIN_TXT and VALID_TXT env vars if data paths differ."
TRAIN_TXT="${TRAIN_TXT:-$PROJECT_DIR/data/TinyStoriesV2-GPT4-train.txt}"
VALID_TXT="${VALID_TXT:-$PROJECT_DIR/data/TinyStoriesV2-GPT4-valid.txt}"

python "$PROJECT_DIR/train.py" \
	--train_txt "$TRAIN_TXT" \
	--valid_txt "$VALID_TXT" \
	--out_dir "$PROJECT_DIR/out" \
	--vocab_size 4000 \
	--d_model 256 \
	--n_heads 8 \
	--n_layers 5 \
	--d_ff 768 \
	--context_length 256 \
	--batch_size 64 \
	--learning_rate 3e-4 \
	--weight_decay 0.1 \
	--warmup_steps 500 \
	--max_steps 10000 \
	--eval_interval 250 \
	--eval_batches 50 \
	--save_interval 1000 \
	--grad_clip_norm 1.0 \
	--use_amp \
	--max_train_stories 1000000 \
	--max_valid_stories 10000 \
	--seed 42 \
	--num_samples 5 \
	"$@"
