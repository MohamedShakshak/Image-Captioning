"""One-time encoder pass — caches pooled 2048-d features as fp16 .npy.

Reads COCO 2017 annotations to discover all train+val images,
then runs the frozen ResNet50 over each and saves fp16 .npy files.

Usage:
    python scripts/cache_features.py \
        --coco-root /kaggle/input/coco-2017 \
        --annotation-train /kaggle/input/coco-2017/annotations/captions_train2017.json \
        --annotation-val /kaggle/input/coco-2017/annotations/captions_val2017.json \
        --out /kaggle/working/features
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from data.transforms import inference_transforms
from models.encoder import Encoder


def _load_image_ids(annotation_path: str, split: str) -> list[dict]:
    """Load image entries from a COCO annotation JSON, tag with split name."""
    with open(annotation_path) as f:
        data = json.load(f)
    for img in data["images"]:
        img["split"] = split
    return data["images"]


def _image_path(coco_root: str, file_name: str) -> Path:
    """COCO 2017 images are in train2017/ or val2017/ based on the file_name prefix.
    Actually all are numeric, so we scan both dirs.
    """
    for sub in ("train2017", "val2017"):
        p = Path(coco_root) / sub / file_name
        if p.exists():
            return p
    return Path(coco_root) / "train2017" / file_name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-root", required=True)
    ap.add_argument("--annotation-train", required=True)
    ap.add_argument("--annotation-val", required=True)
    ap.add_argument("--out", default="features/")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    images = _load_image_ids(args.annotation_train, "train")
    images += _load_image_ids(args.annotation_val, "val")
    print(f"Found {len(images)} images total")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder().to(device).eval()
    transform = inference_transforms()

    skipped = 0
    with torch.no_grad():
        for entry in tqdm(images):
            img_id = entry["id"]
            fname = f"{img_id:012d}"  # 000000435142
            out_path = out / f"{fname}.npy"
            if out_path.exists():
                continue

            file_name = entry["file_name"]
            img_path = Path(args.coco_root) / "train2017" / file_name
            if not img_path.exists():
                img_path = Path(args.coco_root) / "val2017" / file_name
            if not img_path.exists():
                skipped += 1
                continue

            img = Image.open(img_path).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            feat = encoder(x).squeeze(0).cpu().to(torch.float16).numpy()
            np.save(out_path, feat)

    print(f"Done. {len(images) - skipped} features saved in {out} ({skipped} skipped)")

    # Save a splits reference for the dataset
    splits = {}
    for entry in images:
        key = f"{entry['id']:012d}"
        splits[key] = entry["split"]
    (out / "image_splits.json").write_text(json.dumps(splits))
    print(f"Saved image_splits.json ({len(splits)} entries)")


if __name__ == "__main__":
    main()
