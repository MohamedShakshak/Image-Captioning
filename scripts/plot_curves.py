"""Render training curves from metrics.json for the README / release."""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="metrics.json")
    ap.add_argument("--out", default="curves.png")
    args = ap.parse_args()

    with open(args.metrics) as f:
        log = json.load(f)

    epochs = [r["epoch"] for r in log]
    train = [r["train_loss"] for r in log]
    val = [r["val_loss"] for r in log]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train, label="train loss", marker="o")
    ax.plot(epochs, val, label="val loss", marker="s")
    ax.set_xlabel("epoch")
    ax.set_ylabel("masked cross-entropy (label-smoothed)")
    ax.legend()
    ax.set_title("COCO training loss (ResNet50 frozen encoder + LSTM)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
