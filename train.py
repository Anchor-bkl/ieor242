import argparse
import csv
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


@dataclass
class TrainConfig:
    train_txt: str
    valid_txt: str
    out_dir: str
    vocab_size: int
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    context_length: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    max_steps: int
    eval_interval: int
    eval_batches: int
    save_interval: int
    grad_clip_norm: float
    use_amp: bool
    max_train_stories: int
    max_valid_stories: int
    seed: int
    num_samples: int
    sample_prompt: str
    temperature: float
    top_k: int
    top_p: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def line_iterator(path: str, max_lines: int) -> Iterable[str]:
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield line
            count += 1
            if max_lines > 0 and count >= max_lines:
                return


def build_or_load_tokenizer(train_txt: str, tokenizer_path: Path, vocab_size: int, max_stories: int) -> Tokenizer:
    if tokenizer_path.exists():
        return Tokenizer.from_file(str(tokenizer_path))

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
    )

    iterator = line_iterator(train_txt, max_stories)
    tokenizer.train_from_iterator(iterator, trainer=trainer)
    tokenizer.save(str(tokenizer_path))
    return tokenizer


def tokenize_lines(tokenizer: Tokenizer, txt_path: str, eos_id: int, max_stories: int) -> torch.Tensor:
    all_ids: List[int] = []
    for story in tqdm(line_iterator(txt_path, max_stories), desc=f"Tokenizing {Path(txt_path).name}"):
        ids = tokenizer.encode(story).ids
        all_ids.extend(ids)
        all_ids.append(eos_id)
    if not all_ids:
        raise RuntimeError(f"No tokens produced from {txt_path}")
    return torch.tensor(all_ids, dtype=torch.long)


def make_xy(token_ids: torch.Tensor, context_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    n_tokens = token_ids.numel()
    if n_tokens < context_length + 1:
        raise RuntimeError("Not enough tokens for one training chunk.")

    n_chunks = (n_tokens - 1) // context_length
    cut = n_chunks * context_length + 1
    token_ids = token_ids[:cut]

    x = token_ids[:-1].view(n_chunks, context_length)
    y = token_ids[1:].view(n_chunks, context_length)
    return x, y


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)].unsqueeze(0)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + self.dropout(attn_out)
        h = self.ln2(x)
        x = x + self.dropout(self.ffn(h))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, d_ff: int, context_length: int) -> None:
        super().__init__()
        self.context_length = context_length
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, context_length)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.apply(self._init_weights)

        # Weight tying is standard for language modeling and reduces parameters.
        self.lm_head.weight = self.token_emb.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            nn.init.normal_(module.in_proj_weight, mean=0.0, std=0.02)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)
            nn.init.normal_(module.out_proj.weight, mean=0.0, std=0.02)
            if module.out_proj.bias is not None:
                nn.init.zeros_(module.out_proj.bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seqlen = x.shape
        if seqlen > self.context_length:
            raise ValueError("Sequence length exceeds model context_length")

        h = self.token_emb(x)
        h = self.pos_enc(h)

        causal_mask = torch.triu(
            torch.ones(seqlen, seqlen, device=x.device, dtype=torch.bool),
            diagonal=1,
        )

        for block in self.blocks:
            h = block(h, causal_mask)

        h = self.ln_f(h)
        logits = self.lm_head(h)

        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        return logits, loss


def cosine_lr(step: int, base_lr: float, warmup_steps: int, max_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, eval_batches: int, device: torch.device) -> Tuple[float, float]:
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= eval_batches:
            break
        x = x.to(device)
        y = y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    mean_loss = float(np.mean(losses)) if losses else float("nan")
    ppl = float(math.exp(mean_loss)) if math.isfinite(mean_loss) else float("inf")
    model.train()
    return mean_loss, ppl


@torch.no_grad()
def sample_text(
    model: nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 120,
    temperature: float = 1.0,
    top_k: int = 40,
    top_p: float = 0.9,
) -> str:
    model.eval()
    ids = tokenizer.encode(prompt).ids
    x = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        x_cond = x[:, -model.context_length :]
        logits, _ = model(x_cond)
        next_logits = logits[:, -1, :] / max(temperature, 1e-5)

        if top_k > 0:
            v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < v[:, [-1]]] = -float("inf")

        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumprobs = torch.cumsum(probs, dim=-1)
            to_remove = cumprobs > top_p
            to_remove[:, 1:] = to_remove[:, :-1].clone()
            to_remove[:, 0] = False
            sorted_logits[to_remove] = -float("inf")
            next_logits = torch.full_like(next_logits, -float("inf"))
            next_logits.scatter_(1, sorted_indices, sorted_logits)

        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)

    out_ids = x[0].tolist()
    return tokenizer.decode(out_ids)


def write_metrics_row(path: Path, row: List) -> None:
    new_file = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["step", "split", "loss", "ppl", "lr", "tokens_per_sec", "elapsed_sec"])
        writer.writerow(row)


@torch.no_grad()
def save_embedding_neighbors(model: TinyGPT, tokenizer: Tokenizer, out_path: Path, top_k: int = 8) -> None:
    emb = F.normalize(model.token_emb.weight.detach().cpu(), dim=1)
    vocab = tokenizer.get_vocab()
    inv_vocab = {idx: tok for tok, idx in vocab.items()}

    def pick_token(base: str) -> str:
        candidates = [
            base,
            " " + base,
            "Ġ" + base,
            base.capitalize(),
            " " + base.capitalize(),
            "Ġ" + base.capitalize(),
        ]
        for cand in candidates:
            if cand in vocab:
                return cand
        return ""

    def nearest_for(token: str) -> dict:
        tok_id = vocab[token]
        sims = emb @ emb[tok_id]
        values, indices = torch.topk(sims, k=min(top_k + 1, emb.size(0)))
        neighbors = []
        for score, idx in zip(values.tolist(), indices.tolist()):
            if idx == tok_id:
                continue
            neighbors.append({"token": inv_vocab.get(idx, f"<id:{idx}>"), "cosine": float(score)})
            if len(neighbors) >= top_k:
                break
        return {"token_id": tok_id, "neighbors": neighbors}

    def arithmetic(a: str, b: str, c: str) -> list:
        ids = [vocab[a], vocab[b], vocab[c]]
        q = F.normalize(emb[ids[0]] - emb[ids[1]] + emb[ids[2]], dim=0)
        sims = emb @ q
        values, indices = torch.topk(sims, k=min(top_k + 8, emb.size(0)))
        banned = set(ids)
        out = []
        for score, idx in zip(values.tolist(), indices.tolist()):
            if idx in banned:
                continue
            out.append({"token": inv_vocab.get(idx, f"<id:{idx}>"), "cosine": float(score)})
            if len(out) >= top_k:
                break
        return out

    base_picks = ["boy", "girl", "dog", "cat", "bird", "dad", "mom"]
    resolved = {base: pick_token(base) for base in base_picks}
    results = {"resolved_tokens": resolved, "neighbors": {}, "arithmetic": {}}

    for base, token in resolved.items():
        if not token:
            continue
        info = nearest_for(token)
        info["matched_token"] = token
        results["neighbors"][base] = info

    arithmetic_specs = [
        ("girl", "boy", "dog"),
        ("mom", "dad", "boy"),
        ("cat", "dog", "bird"),
    ]
    for a, b, c in arithmetic_specs:
        if not resolved.get(a) or not resolved.get(b) or not resolved.get(c):
            continue
        key = f"{a}-{b}+{c}"
        results["arithmetic"][key] = {
            "tokens": [resolved[a], resolved[b], resolved[c]],
            "top": arithmetic(resolved[a], resolved[b], resolved[c]),
        }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--train_txt", type=str, required=True)
    p.add_argument("--valid_txt", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="out")
    p.add_argument("--vocab_size", type=int, default=4000)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_layers", type=int, default=5)
    p.add_argument("--d_ff", type=int, default=768)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--eval_batches", type=int, default=50)
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--max_train_stories", type=int, default=1000000)
    p.add_argument("--max_valid_stories", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_samples", type=int, default=5)
    p.add_argument("--sample_prompt", type=str, default="Once upon a time,")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--top_p", type=float, default=0.9)

    a = p.parse_args()
    return TrainConfig(**vars(a))


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    host = os.uname().nodename
    if not os.environ.get("SLURM_JOB_ID") and host.startswith("br"):
        raise RuntimeError(
            "Refusing to train on login node. Use salloc/sbatch to run on a compute node."
        )

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer_path = out_dir / "tokenizer.json"
    tokenizer = build_or_load_tokenizer(
        train_txt=cfg.train_txt,
        tokenizer_path=tokenizer_path,
        vocab_size=cfg.vocab_size,
        max_stories=cfg.max_train_stories,
    )
    actual_vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    print(f"Tokenizer vocab size: requested={cfg.vocab_size}, actual={actual_vocab_size}")

    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        raise RuntimeError("Tokenizer missing <eos> token")

    train_ids = tokenize_lines(tokenizer, cfg.train_txt, eos_id=eos_id, max_stories=cfg.max_train_stories)
    valid_ids = tokenize_lines(tokenizer, cfg.valid_txt, eos_id=eos_id, max_stories=cfg.max_valid_stories)

    x_train, y_train = make_xy(train_ids, cfg.context_length)
    x_valid, y_valid = make_xy(valid_ids, cfg.context_length)

    print(f"Train tokens: {train_ids.numel()}, chunks: {x_train.size(0)}")
    print(f"Valid tokens: {valid_ids.numel()}, chunks: {x_valid.size(0)}")

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(TensorDataset(x_valid, y_valid), batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    model = TinyGPT(
        vocab_size=actual_vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
        context_length=cfg.context_length,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.use_amp and device.type == "cuda"))

    metrics_path = out_dir / "train_metrics.csv"

    # Initial validation sanity check.
    init_val_loss, init_val_ppl = evaluate(model, valid_loader, cfg.eval_batches, device)
    print(f"Initial valid loss={init_val_loss:.4f}, ppl={init_val_ppl:.2f}")
    write_metrics_row(metrics_path, [0, "valid", init_val_loss, init_val_ppl, 0.0, 0.0, 0.0])

    step = 0
    start = time.time()
    train_iter = iter(train_loader)

    while step < cfg.max_steps:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x = x.to(device)
        y = y.to(device)

        lr = cosine_lr(step, cfg.learning_rate, cfg.warmup_steps, cfg.max_steps)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)

        autocast_enabled = cfg.use_amp and device.type == "cuda"
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if autocast_enabled
            else nullcontext()
        )
        with amp_ctx:
            _, loss = model(x, y)

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        scaler.step(opt)
        scaler.update()

        elapsed = time.time() - start
        tokens_seen = (step + 1) * cfg.batch_size * cfg.context_length
        tok_per_sec = tokens_seen / max(elapsed, 1e-6)

        if (step + 1) % 20 == 0:
            train_loss = float(loss.item())
            train_ppl = float(math.exp(train_loss)) if train_loss < 20 else float("inf")
            print(
                f"step={step+1} train_loss={train_loss:.4f} train_ppl={train_ppl:.2f} "
                f"lr={lr:.6g} tok/s={tok_per_sec:.1f}"
            )
            write_metrics_row(metrics_path, [step + 1, "train", train_loss, train_ppl, lr, tok_per_sec, elapsed])

        if (step + 1) % cfg.eval_interval == 0:
            val_loss, val_ppl = evaluate(model, valid_loader, cfg.eval_batches, device)
            print(f"step={step+1} valid_loss={val_loss:.4f} valid_ppl={val_ppl:.2f}")
            write_metrics_row(metrics_path, [step + 1, "valid", val_loss, val_ppl, lr, tok_per_sec, elapsed])

        if (step + 1) % cfg.save_interval == 0:
            ckpt = {
                "step": step + 1,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "config": vars(cfg),
            }
            ckpt_path = out_dir / f"checkpoint_step_{step+1}.pt"
            torch.save(ckpt, ckpt_path)

            samples_path = out_dir / f"samples_step_{step+1}.txt"
            with open(samples_path, "w", encoding="utf-8") as f:
                f.write(f"Prompt: {cfg.sample_prompt}\n")
                f.write("=" * 60 + "\n")
                for i in range(cfg.num_samples):
                    text = sample_text(
                        model,
                        tokenizer,
                        prompt=cfg.sample_prompt,
                        device=device,
                        max_new_tokens=120,
                        temperature=cfg.temperature,
                        top_k=cfg.top_k,
                        top_p=cfg.top_p,
                    )
                    f.write(f"\n--- Sample {i+1} ---\n{text}\n")

        step += 1

    final_ckpt = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "config": vars(cfg),
    }
    torch.save(final_ckpt, out_dir / "checkpoint_final.pt")

    save_embedding_neighbors(model, tokenizer, out_dir / "embedding_neighbors.json", top_k=8)

    val_loss, val_ppl = evaluate(model, valid_loader, cfg.eval_batches, device)
    elapsed = time.time() - start
    tok_per_sec = (step * cfg.batch_size * cfg.context_length) / max(elapsed, 1e-6)
    write_metrics_row(metrics_path, [step, "valid_final", val_loss, val_ppl, cfg.learning_rate, tok_per_sec, elapsed])

    print("Training complete")
    print(f"Final valid loss={val_loss:.4f}, ppl={val_ppl:.2f}")
    print(f"Total elapsed={elapsed:.1f}s, tokens/sec={tok_per_sec:.1f}")


if __name__ == "__main__":
    main()
