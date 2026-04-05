import csv
from pathlib import Path


def read_rows(csv_path: Path):
    with csv_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def summarize_one(exp_dir: Path):
    metrics = exp_dir / "train_metrics.csv"
    if not metrics.exists():
        return None

    rows = read_rows(metrics)
    valid_rows = [r for r in rows if r["split"].startswith("valid")]
    train_rows = [r for r in rows if r["split"] == "train"]

    if not valid_rows:
        return None

    best_valid = min(valid_rows, key=lambda r: to_float(r["loss"]))
    final_valid = valid_rows[-1]
    final_train = train_rows[-1] if train_rows else None

    return {
        "experiment": exp_dir.name,
        "best_valid_loss": to_float(best_valid["loss"]),
        "best_valid_ppl": to_float(best_valid["ppl"]),
        "best_valid_step": int(best_valid["step"]),
        "final_valid_loss": to_float(final_valid["loss"]),
        "final_valid_ppl": to_float(final_valid["ppl"]),
        "final_step": int(final_valid["step"]),
        "final_tokens_per_sec": to_float(final_valid["tokens_per_sec"]),
        "final_elapsed_sec": to_float(final_valid["elapsed_sec"]),
        "final_train_loss": to_float(final_train["loss"]) if final_train else float("nan"),
        "final_train_ppl": to_float(final_train["ppl"]) if final_train else float("nan"),
    }


def main():
    root = Path("experiments")
    out_csv = root / "summary.csv"
    out_md = root / "summary.md"

    exp_dirs = sorted([p for p in root.iterdir() if p.is_dir()]) if root.exists() else []
    records = []
    for exp_dir in exp_dirs:
        rec = summarize_one(exp_dir)
        if rec is not None:
            records.append(rec)

    if not records:
        print("No completed experiment metrics found under experiments/*/train_metrics.csv")
        return

    fields = list(records[0].keys())
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)

    lines = []
    lines.append("| experiment | best_valid_loss | best_valid_ppl | best_valid_step | final_valid_loss | final_valid_ppl | final_tokens_per_sec | final_elapsed_sec |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in records:
        lines.append(
            f"| {r['experiment']} | {r['best_valid_loss']:.4f} | {r['best_valid_ppl']:.2f} | {r['best_valid_step']} | {r['final_valid_loss']:.4f} | {r['final_valid_ppl']:.2f} | {r['final_tokens_per_sec']:.1f} | {r['final_elapsed_sec']:.1f} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
