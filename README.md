# TinyStories Decoder-Only Transformer (HW3)

This repository contains a complete TinyStories homework project:
- training pipeline (`train.py`)
- generation script (`generate.py`)
- embedding analysis (`analyze_embeddings.py`)
- ablation/experiment summaries (`experiments/`)
- final report notebook (`HOMEWORK_REPORT.ipynb`)

## Quick Start: Inference From Uploaded Final Checkpoint

The repository includes a ready-to-use final checkpoint for the main model:
- `out_main_gpu_baseline/checkpoint_final.pt`
- `out_main_gpu_baseline/tokenizer.json`

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Run generation directly (no retraining required):

```bash
python generate.py \
	--checkpoint out_main_gpu_baseline/checkpoint_final.pt \
	--tokenizer out_main_gpu_baseline/tokenizer.json \
	--prompt "Once upon a time," \
	--num_samples 5 \
	--max_new_tokens 120 \
	--temperature 0.9 \
	--top_k 40 \
	--top_p 0.9
```

## Main Training Setup (Used For Final Results)

- Model: decoder-only transformer
- Config: `vocab_size=4000`, `d_model=256`, `n_heads=8`, `n_layers=5`, `d_ff=768`, `context_length=256`
- Data subset: first `1,000,000` train stories and first `10,000` valid stories
- Main output directory: `out_main_gpu_baseline/`

## Reproduce Training (HPC)

This project is designed for SLURM compute nodes. `run.sh` blocks accidental login-node training.

Batch:

```bash
sbatch submit_main_gpu_baseline.slurm
```

Interactive:

```bash
salloc -A mth250011p -p GPU-shared --gpus=1 -N 1 -t 04:00
bash run.sh --out_dir out_main_gpu_baseline
```

## Important Files

- `HOMEWORK_REPORT.ipynb`: final report with plots/tables and appendix code
- `ASSIGNMENT_SUMMARY.md`: concise project summary
- `experiments/summary.csv`: quantitative comparison across ablations
- `out_main_gpu_baseline/train_metrics.csv`: main run training/validation curves data

## Notes

- Raw dataset files and intermediate checkpoints are excluded from GitHub to keep the repository lightweight.
- The final inference checkpoint is included for direct reproducibility.
