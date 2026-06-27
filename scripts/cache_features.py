"""One-time encoder pass — caches pooled 2048-d features as fp16 .npy.

    python scripts/cache_features.py --coco-root /kaggle/input/coco-2017 \
        --karpathy-split /kaggle/input/coco-karpathy/dataset_coco.json \
        --out /kaggle/working/features

See PLAN.md § Image Preprocessing (uses the inference transform — deterministic).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from data.transforms import inference_transforms
from models.encoder import Encoder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-root", required=True)
    ap.add_argument("--karpathy-split", required=True)
    ap.add_argument("--out", default="features/")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    import json

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with open(args.karpathy_split) as f:
        split = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder().to(device).eval()
    transform = inference_transforms()

    images = [i for i in split["images"] if i["split"] in {"train", "val", "test"}]
    with torch.no_grad():
        for entry in tqdm(images):
            img_id = entry["filename"].split(".")[0]
            out_path = out / f"{img_id}.npy"
            if out_path.exists():
                continue
            img_path = Path(args.coco_root) / entry["filepath"] / entry["filename"]
            img = Image.open(img_path).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            feat = encoder(x).squeeze(0).cpu().to(torch.float16).numpy()
            np.save(out_path, feat)

    print(f"Done. Cached features in {out}")


if __name__ == "__main__":
    main()
