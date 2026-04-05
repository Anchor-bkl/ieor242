import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from train import TinyGPT, sample_text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate text from a trained TinyStories model checkpoint")
    p.add_argument("--checkpoint", type=str, default="out/checkpoint_final.pt")
    p.add_argument("--tokenizer", type=str, default="out/tokenizer.json")
    p.add_argument("--prompt", type=str, default="Once upon a time,")
    p.add_argument("--num_samples", type=int, default=5)
    p.add_argument("--max_new_tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()

    ckpt_path = Path(args.checkpoint)
    tok_path = Path(args.tokenizer)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tok_path}")

    device = pick_device(args.device)
    print(f"Using device: {device}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    cfg = checkpoint["config"]

    tokenizer = Tokenizer.from_file(str(tok_path))
    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)

    model = TinyGPT(
        vocab_size=vocab_size,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        context_length=cfg["context_length"],
    ).to(device)

    model.load_state_dict(checkpoint["model"])
    model.eval()

    print(f"Loaded checkpoint step: {checkpoint.get('step', 'unknown')}")
    print(f"Prompt: {args.prompt}")
    print("=" * 72)

    for i in range(args.num_samples):
        text = sample_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        print(f"\n--- Sample {i + 1} ---")
        print(text)


if __name__ == "__main__":
    main()
