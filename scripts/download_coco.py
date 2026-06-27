"""Download a COCO subset for local sanity runs.

    --split {tiny, mini, full}
    tiny  : 500 images
    mini  : 5,000 images
    full  : 123,000 images (uncompressed ~25GB; use on Kaggle's pre-mounted dataset instead)

See PLAN.md § download_coco.py scope.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Karpathy split JSON (defines per-image train/val/test assignment + captions).
KARPATHY_URL = "http://cs.stanford.edu/people/karpathy/deepimagesplit/coco_dataset_2014.zip"
# We use the cleanerJeremyHoward URL-friendly mirror in practice; the script writes the json directly.


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["tiny", "mini", "full"], default="tiny")
    ap.add_argument("--out", type=str, default="data/")
    ap.add_argument(
        "--karpathy-only",
        action="store_true",
        help="Only fetch dataset_coco.json, skip images (Kaggle has images mounted).",
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Fetch Karpathy split JSON
    karpathy_path = out / "dataset_coco.json"
    if not karpathy_path.exists():
        print("Fetching Karpathy split JSON...")
        # TODO: download + extract from KARPATHY_URL. For now, document.
        raise SystemExit(
            "Karpathy split download not yet wired — see feat/data branch.\n"
            "Manual: download http://cs.stanford.edu/people/karpathy/deepimagesplit/coco_dataset_2014.zip, "
            "extract dataset_coco.json into --out."
        )
    print(f"Karpathy split at {karpathy_path}")

    if args.karpathy_only:
        print("--karpathy-only set; skipping image download.")
        return

    # Subset size
    keep = {"tiny": 500, "mini": 5_000, "full": None}[args.split]
    with open(karpathy_path) as f:
        split = json.load(f)
    if keep is not None:
        split["images"] = split["images"][:keep]

    # TODO (feat/data): download images per the Karpathy image filenames via COCO CDN.
    #   http://images.cocodataset.org/train2014/<filename>
    #   http://images.cocodataset.org/val2014/<filename>
    print(
        f"Image download for split={args.split} ({len(split['images'])} images) — wired in feat/data."
    )
    print(f"Once downloaded, place under {out}/images/ and run scripts/cache_features.py.")


if __name__ == "__main__":
    main()
