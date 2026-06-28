"""Standalone Streamlit demo for HuggingFace Spaces.

Self-contained: no dependency on the src/ package.
Downloads weights from HF Hub at startup.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PAD_IDX = 0
UNK_IDX = 3
END_IDX = 2
HF_REPO_ENV = os.environ.get("HF_REPO", "MohamedShakshak/image-captioning-pytorch")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Beam:
    tokens: list[int]
    score: float


@dataclass
class CaptionResult:
    text: str
    beams: list[Beam] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


class Vocab:
    def __init__(self, word2idx: dict[str, int]) -> None:
        self.word2idx = word2idx
        self.idx2word = {i: w for w, i in word2idx.items()}

    def decode(self, idxs: list[int]) -> list[str]:
        return [self.idx2word.get(i, "<unk>") for i in idxs]

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        return cls(json.loads(Path(path).read_text())["word2idx"])


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class Encoder(nn.Module):
    """Frozen ResNet50 without final FC."""

    def __init__(self) -> None:
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.features(images).flatten(1)


class Decoder(nn.Module):
    """Single-layer LSTM with image-conditioned initial states."""

    def __init__(self, vocab_size: int, hidden_dim: int = 512, embedding_dim: int = 512) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.init_h = nn.Linear(2048, hidden_dim)
        self.init_c = nn.Linear(2048, hidden_dim)

    def forward(self, features: torch.Tensor, caption_idxs: torch.Tensor) -> torch.Tensor:
        h0 = torch.tanh(self.init_h(features)).unsqueeze(0)
        c0 = torch.tanh(self.init_c(features)).unsqueeze(0)
        emb = self.embed(caption_idxs)
        out, _ = self.lstm(emb, (h0, c0))
        return self.head(out)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@torch.no_grad()
def beam_search(
    decoder: Decoder,
    features: torch.Tensor,
    beam_size: int = 3,
    max_len: int = 20,
    length_norm_alpha: float = 0.7,
) -> list[tuple[list[int], float]]:
    """Beam search (single image). Returns list of (tokens, score) sorted desc."""
    device = features.device
    h = torch.tanh(decoder.init_h(features)).unsqueeze(0)
    c = torch.tanh(decoder.init_c(features)).unsqueeze(0)

    beams: list[tuple[torch.Tensor, float, torch.Tensor, torch.Tensor]] = [
        (torch.tensor([1], device=device), 0.0, h, c)
    ]
    finished: list[tuple[list[int], float]] = []

    for _ in range(max_len):
        candidates: list[tuple[torch.Tensor, float, torch.Tensor, torch.Tensor]] = []
        for tokens, score, hh, cc in beams:
            x = tokens[-1:].unsqueeze(0)
            emb = decoder.embed(x)
            out, (nh, nc) = decoder.lstm(emb, (hh, cc))
            logits = decoder.head(out[:, -1, :])
            logits[:, UNK_IDX] = float("-inf")
            logits[:, PAD_IDX] = float("-inf")
            log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)
            top_lp, top_idx = log_probs.topk(beam_size)
            for i in range(beam_size):
                tok = int(top_idx[i].item())
                lp = float(top_lp[i].item())
                new_tokens = torch.cat([tokens, torch.tensor([tok], device=device)])
                candidates.append((new_tokens, score + lp, nh, nc))

        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_size]

        alive: list[tuple[torch.Tensor, float, torch.Tensor, torch.Tensor]] = []
        for tokens, score, h_n, c_n in beams:
            if tokens[-1].item() == END_IDX:
                length = tokens.numel() - 1
                norm_score = score / (length ** length_norm_alpha)
                finished.append((tokens[1:-1].tolist(), norm_score))
            else:
                alive.append((tokens, score, h_n, c_n))
        beams = alive
        if not beams:
            break

    if not finished:
        if beams:
            tokens, score, *_ = beams[0]
        else:
            tokens, score = torch.tensor([1, 2], device=device), 0.0
        length = max(1, tokens.numel() - 1)
        norm_score = score / (length ** length_norm_alpha)
        finished.append((tokens[1:].tolist(), norm_score))

    finished.sort(key=lambda x: x[1], reverse=True)
    return finished


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


@st.cache_resource
def load_model() -> tuple[Encoder, Decoder, Vocab]:
    ckpt = torch.load(hf_hub_download(HF_REPO_ENV, "best.pt"), map_location="cpu")
    vocab = Vocab.load(hf_hub_download(HF_REPO_ENV, "vocab.json"))
    decoder = Decoder(vocab_size=len(vocab.word2idx))
    decoder.load_state_dict(ckpt["model"])
    encoder = Encoder()
    return encoder, decoder, vocab


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Image Captioning", layout="centered")
st.title("Image Captioning")
st.markdown("ResNet50 encoder · LSTM decoder · Beam search k=3")

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.stop()

img = Image.open(uploaded).convert("RGB")
st.image(img, caption="Input", use_container_width=True)

with st.spinner("Generating caption..."):
    encoder, decoder, vocab = load_model()
    x = transform(img).unsqueeze(0)
    feats = encoder(x).squeeze(0)
    results = beam_search(decoder, feats)

st.subheader("Caption")
top_tokens, _ = results[0]
st.markdown(f"### {' '.join(vocab.decode(top_tokens))}")

if len(results) > 1:
    st.subheader("Beam search")
    for i, (tokens_raw, score) in enumerate(results):
        tokens_str = " ".join(vocab.decode(tokens_raw))
        st.markdown(f"**Beam {i + 1}** (score: `{score:.4f}`): {tokens_str}")