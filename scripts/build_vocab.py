"""Build the word vocabulary from COCO 2017 captions.

    python scripts/build_vocab.py \
        --annotation-train /kaggle/input/coco-2017/annotations/captions_train2017.json \
        --annotation-val /kaggle/input/coco-2017/annotations/captions_val2017.json \
        --min-freq 5 --out vocab.json
"""

from __future__ import annotations

import argparse
import json

from data.vocab import Vocab


def _load_captions(annotation_path: str) -> list[str]:
    with open(annotation_path) as f:
        data = json.load(f)
    return [ann["caption"] for ann in data["annotations"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotation-train", required=True)
    ap.add_argument("--annotation-val", required=True)
    ap.add_argument("--min-freq", type=int, default=5)
    ap.add_argument("--out", default="vocab.json")
    args = ap.parse_args()

    captions = _load_captions(args.annotation_train)
    captions += _load_captions(args.annotation_val)
    print(f"Loaded {len(captions)} captions total")

    vocab = Vocab.build(captions, min_freq=args.min_freq)
    vocab.save(args.out)
    print(f"Saved vocab ({vocab.size} words) to {args.out}")


if __name__ == "__main__":
    main()
