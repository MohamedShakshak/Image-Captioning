"""Build the word vocabulary from COCO Karpathy train captions.

    python scripts/build_vocab.py \
        --karpathy-split /kaggle/input/coco-karpathy/dataset_coco.json \
        --min-freq 5 --out vocab.json

Saves vocab.json with word2idx + special token indices (see image_captioning.data.vocab).
"""

from __future__ import annotations

import argparse
import json

from data.vocab import Vocab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--karpathy-split", required=True)
    ap.add_argument("--min-freq", type=int, default=5)
    ap.add_argument("--out", default="vocab.json")
    args = ap.parse_args()

    with open(args.karpathy_split) as f:
        split = json.load(f)

    captions = []
    for entry in split["images"]:
        if entry["split"] != "train":
            continue
        for s in entry["sentences"]:
            captions.append(s["raw"])

    vocab = Vocab.build(captions, min_freq=args.min_freq)
    vocab.save(args.out)
    print(f"Saved vocab ({vocab.size} words) to {args.out}")


if __name__ == "__main__":
    main()
