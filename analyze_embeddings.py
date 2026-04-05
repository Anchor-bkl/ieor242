import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from train import TinyGPT, save_embedding_neighbors


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run embedding-space analysis for assignment part (f)")
    p.add_argument("--checkpoint", type=str, default="out/checkpoint_final.pt")
    p.add_argument("--tokenizer", type=str, default="out/tokenizer.json")
    p.add_argument("--out_json", type=str, default="out/embedding_neighbors.json")
    p.add_argument("--top_k", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ckpt_path = Path(args.checkpoint)
    tok_path = Path(args.tokenizer)
    out_path = Path(args.out_json)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tok_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
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
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    save_embedding_neighbors(model, tokenizer, out_path, top_k=args.top_k)

    data = json.loads(out_path.read_text(encoding="utf-8"))
    print(f"Saved embedding analysis to: {out_path}")
    print("Resolved tokens:")
    for k, v in data.get("resolved_tokens", {}).items():
        if v:
            print(f"  {k} -> {v}")

    print("\nNeighbor queries available:")
    for k in data.get("neighbors", {}).keys():
        print(f"  {k}")

    print("\nArithmetic queries available:")
    for k in data.get("arithmetic", {}).keys():
        print(f"  {k}")


if __name__ == "__main__":
    main()
