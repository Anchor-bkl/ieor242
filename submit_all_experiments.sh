#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$PROJECT_DIR/experiments_logs"
EXP_DIR="$PROJECT_DIR/experiments"
mkdir -p "$LOG_DIR" "$EXP_DIR"

ACCOUNT="mth250011p"
PARTITION="GPU-shared"
TIME_LIMIT="02:00:00"
BASELINE_ARGS="--vocab_size 4000 --d_model 256 --n_heads 8 --n_layers 5 --d_ff 768 --context_length 256 --batch_size 64 --learning_rate 3e-4 --weight_decay 0.1 --warmup_steps 500 --max_steps 10000 --eval_interval 250 --save_interval 1000 --eval_batches 50 --max_train_stories 1000000 --max_valid_stories 10000 --num_samples 5"

submit_exp() {
  local name="$1"
  local args="$2"
  local out_dir="$EXP_DIR/$name"
  mkdir -p "$out_dir"

  sbatch \
    -A "$ACCOUNT" \
    -p "$PARTITION" \
    -N 1 \
    --gpus=1 \
    -t "$TIME_LIMIT" \
    -J "$name" \
    -o "$LOG_DIR/%x-%j.out" \
    --wrap="cd '$PROJECT_DIR' && bash run.sh --out_dir '$out_dir' $BASELINE_ARGS $args"
}

# Baseline shared by all experiment groups.
  submit_exp "exp_baseline" ""

# (g-i) Effect of model size
submit_exp "exp_model_small" "--d_model 64 --n_heads 4 --n_layers 4 --d_ff 192"
submit_exp "exp_model_large" "--d_model 256 --n_heads 8 --n_layers 4 --d_ff 768"

# (g-ii) Effect of vocabulary size
submit_exp "exp_vocab_1000" "--vocab_size 1000"
submit_exp "exp_vocab_5000" "--vocab_size 5000"

# (g-iii) Effect of context length
submit_exp "exp_ctx_64" "--context_length 64"
submit_exp "exp_ctx_128" "--context_length 128"

squeue -u "$USER"
