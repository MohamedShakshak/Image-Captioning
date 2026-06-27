"""COCO dataset and collate_fn for the cached-features training pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .vocab import Vocab, encode_caption, END_IDX, PAD_IDX


class CachedCOCODataset(Dataset):
    """Loads cached encoder features + per-image captions.

    Each item: (features: Tensor[2048], caption_idxs: list[int])
    """

    def __init__(
        self,
        features_dir: str | Path,
        karpathy_split: dict[str, Any],
        vocab: Vocab,
        split: str = "train",
    ) -> None:
        self.features_dir = Path(features_dir)
        self.vocab = vocab
        self.entries = [
            e for e in karpathy_split["images"] if e["split"] == split
        ]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, list[int]]:
        entry = self.entries[idx]
        img_id = entry["filename"].split(".")[0]
        feat = np.load(self.features_dir / f"{img_id}.npy").astype(np.float32)
        # Use the first of the 5 captions for V1 — multi-caption is a v2 idea.
        caption = entry["sentences"][0]["raw"]
        idxs = encode_caption(self.vocab, caption)
        return torch.from_numpy(feat), idxs


def collate_fn(
    batch: list[tuple[torch.Tensor, list[int]]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad captions to batch-max length. Returns (features, caption_idxs, lengths)."""
    features = torch.stack([b[0] for b in batch])
    lengths = torch.tensor([len(b[1]) for b in batch], dtype=torch.long)
    max_len = int(lengths.max())
    padded = torch.full(
        (len(batch), max_len), PAD_IDX, dtype=torch.long
    )
    for i, (_, idxs) in enumerate(batch):
        padded[i, : len(idxs)] = torch.tensor(idxs, dtype=torch.long)
    return features, padded, lengths