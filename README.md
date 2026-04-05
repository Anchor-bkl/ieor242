# TinyStories GPT Homework Minimal Skeleton

This project is a minimal, compute-node-safe skeleton for your homework.

## 1) Strict remote resource workflow

Do lightweight work on login node only:
- edit code
- prepare environment
- submit jobs

Do training/evaluation on compute node only:
- via interactive allocation (`salloc`)
- or batch job (`sbatch`)

This repo enforces that rule in `run.sh`:
- exits if no `SLURM_JOB_ID`
- exits if hostname looks like login node (starts with `br`)

## 2) Setup (login node)

```bash
cd ~/hw3_tinystories_minimal
python3 -m venv ~/venvs/hw3
source ~/venvs/hw3/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Download data to `data/`:

```bash
mkdir -p data
cd data
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
```

## 3) Run on compute resources

Option A: batch (recommended)

```bash
cd ~/hw3_tinystories_minimal
sbatch submit.slurm
squeue -u "$USER"
```

Option B: interactive

```bash
salloc -A mth250011p -p GPU-shared --gpus=1 -N 1 -t 04:00
cd ~/hw3_tinystories_minimal
bash run.sh
```

## 4) Outputs

All outputs go to `out/`:
- `train_metrics.csv`: step, split, loss, ppl, lr, tokens_per_sec, elapsed_sec
- `samples_step_*.txt`: generated samples
- `checkpoint_step_*.pt`, `checkpoint_final.pt`
- `tokenizer.json`

Use `report_template.ipynb` to generate plots/tables for your writeup.

## 5) Useful overrides

You can override defaults in `run.sh` by passing args:

```bash
bash run.sh --max_steps 2000 --vocab_size 2000 --context_length 128 --batch_size 32
```
