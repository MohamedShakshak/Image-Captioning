"""Streamlit demo — caption an uploaded image. Deploys to HF Spaces (Dockerfile).

Beam search panel shows all k=3 hypotheses with cumulative log-probs.
"""

from __future__ import annotations

import os

import streamlit as st
import torch
from PIL import Image

from captioner import Captioner
from models.encoder import Encoder


@st.cache_resource
def load_captioner() -> Captioner:
    hf_repo = os.environ.get("HF_REPO", "MohamedShakshak/image-captioning-pytorch")
    cap = Captioner.from_pretrained(hf_repo, device="cpu")
    # Attach a live encoder for the demo (cached features are Kaggle-only).
    cap.encoder = Encoder().to(cap.device).eval()
    return cap


def main() -> None:
    st.title("Image Captioning (PyTorch · ResNet50 + LSTM)")
    st.write("Upload an image — the model produces a caption via beam search (k=3).")

    uploaded = st.file_uploader("Image", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        return

    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input", use_column_width=True)

    cap = load_captioner()
    with torch.no_grad():
        feats = cap.encoder(cap.transform(img).unsqueeze(0).to(cap.device)).squeeze(0)
        result = cap.caption_from_features(feats, beam=True)

    st.subheader("Top caption")
    st.write(result.text)

    if result.beams:
        st.subheader(f"Beam search (k={len(result.beams)})")
        for i, b in enumerate(result.beams):
            tokens = " ".join(cap.vocab.decode(b.tokens))
            st.markdown(f"**Beam {i + 1}** (score `{b.score:.4f}`): {tokens}")


if __name__ == "__main__":
    main()