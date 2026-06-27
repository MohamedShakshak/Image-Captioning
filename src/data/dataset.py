"""COCO 2017 dataset + collate_fn for the cached-features training pipeline.

Uses standard COCO 2017 annotation JSONs (captions_train2017.json,
captions_val2017.json) instead of Karpathy split.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .vocab import PAD_IDX, Vocab, encode_caption


class CachedCOCODataset(Dataset):
    """Loads cached encoder features + COCO 2017 captions.

    Each item: (features: Tensor[2048], caption_idxs: list[int])

    Uses only the FIRST caption per image (of the 5 available).
    """

    def __init__(
        self,
        features_dir: str | Path,
        annotation_json: str | Path,
        vocab: Vocab,
    ) -> None:
        self.features_dir = Path(features_dir)
        self.vocab = vocab

        with open(annotation_json) as f:
            data = json.load(f)

        # Build mapping: image_id → first caption
        img_to_caption: dict[int, str] = {}
        for ann in data["annotations"]:
            iid = ann["image_id"]
            if iid not in img_to_caption:
                img_to_caption[iid] = ann["caption"]

        self.entries: list[tuple[int, str]] = sorted(img_to_caption.items())
        self.image_ids = [e[0] for e in self.entries]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[int, torch.Tensor, list[int]]:
        img_id, caption = self.entries[idx]
        fname = f"{img_id:012d}"
        feat = np.load(self.features_dir / f"{fname}.npy").astype(np.float32)
        idxs = encode_caption(self.vocab, caption)
        return img_id, torch.from_numpy(feat), idxs


def collate_fn(
    batch: list[tuple[int, torch.Tensor, list[int]]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad captions to batch-max length. Returns (features, caption_idxs, lengths)."""
    features = torch.stack([b[1] for b in batch])
    lengths = torch.tensor([len(b[2]) for b in batch], dtype=torch.long)
    max_len = int(lengths.max())
    padded = torch.full((len(batch), max_len), PAD_IDX, dtype=torch.long)
    for i, (_, _, idxs) in enumerate(batch):
        padded[i, : len(idxs)] = torch.tensor(idxs, dtype=torch.long)
    return features, padded, lengths
