"""Vocabulary class — build/save/load. Specials: <pad>=0, <start>=1, <end>=2, <unk>=3.

See PLAN.md § Vocabulary."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

PAD_TOKEN = "<pad>"
START_TOKEN = "<start>"
END_TOKEN = "<end>"
UNK_TOKEN = "<unk>"

PAD_IDX = 0
START_IDX = 1
END_IDX = 2
UNK_IDX = 3

SPECIALS = [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN]

# PTB-style contraction split (built-in, no dep).
CONTRACTIONS = {
    "won't": "will not",
    "can't": "can not",
    "n't": " not",
    "'ll": " will",
    "'re": " are",
    "'ve": " have",
    "'m": " am",
    "'d": " would",
    "'s": " is",
}


class Vocab:
    """Word-level vocabulary with reserved special-token indices."""

    def __init__(self, word2idx: dict[str, int]) -> None:
        self.word2idx = word2idx
        self.idx2word = {i: w for w, i in word2idx.items()}

    @property
    def size(self) -> int:
        return len(self.word2idx)

    def __len__(self) -> int:
        return self.size

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.word2idx.get(t, UNK_IDX) for t in tokens]

    def decode(self, idxs: list[int]) -> list[str]:
        return [self.idx2word[i] for i in idxs]

    @classmethod
    def build(cls, captions: list[str], min_freq: int = 5) -> "Vocab":
        counter: Counter[str] = Counter()
        for cap in captions:
            counter.update(tokenize(cap))
        words = [w for w, c in counter.most_common() if c >= min_freq]
        word2idx = {tok: i for i, tok in enumerate(SPECIALS)}
        for w in words:
            if w not in word2idx:
                word2idx[w] = len(word2idx)
        return cls(word2idx)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"word2idx": self.word2idx}, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        data = json.loads(Path(path).read_text())
        return cls(data["word2idx"])


def tokenize(caption: str) -> list[str]:
    """Lowercase, PTB-style contraction split, strip punctuation, whitespace split."""
    s = caption.lower()
    for src, dst in CONTRACTIONS.items():
        s = s.replace(src, dst)
    # Strip punctuation
    out = []
    for tok in s.split():
        t = tok.strip(".,;:!?\"'`()")
        if t:
            out.append(t)
    return out


def encode_caption(vocab: Vocab, caption: str) -> list[int]:
    """Wrap tokenized caption with <start> ... <end>."""
    return [START_IDX, *vocab.encode(tokenize(caption)), END_IDX]