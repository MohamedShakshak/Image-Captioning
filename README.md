# Image Captioning (PyTorch)

PyTorch port of an image captioning model with a **ResNet50 encoder + LSTM decoder**, trained on **COCO 2017**, evaluated with `pycocoevalcap`, and deployed as a **Streamlit demo on HuggingFace Spaces**.

> Vanilla LSTM v1 — attention is a documented v2 milestone.

![CI](https://img.shields.io/github/actions/workflow/status/MohamedShakshak/Image-Captioning/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/github/license/MohamedShakshak/Image-Captioning)
[![Spaces](https://img.shields.io/badge/%F0%9F%A4%97-Spaces-blue)](https://huggingface.co/spaces/MohamedShakshak/image-captioning-pytorch)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97-Model-blue)](https://huggingface.co/MohamedShakshak/image-captioning-pytorch)

## Live demo

A Streamlit app is hosted on HuggingFace Spaces — upload an image, get the top caption plus the k=3 beam search hypotheses with cumulative log-probs.

**[Open the demo](https://huggingface.co/spaces/MohamedShakshak/image-captioning-pytorch)**

## Architecture

```mermaid
flowchart LR
    img[Image] --> enc[ResNet50<br/>frozen encoder]
    enc -->|2048-d pooled vector| cache[(cached .npy<br/>fp16, ~1GB)]
    cache --> h0[h0 = tanh W_h feat]
    cache --> c0[c0 = tanh W_c feat]
    h0 --> dec[LSTM decoder<br/>hidden=512, dropout=0.5]
    c0 --> dec
    start[&lt;start&gt; token] --> dec
    dec -->|tokens| beam[Beam search k=3<br/>length-normed]
    beam --> cap[Top caption + beams]
```

## Results (COCO 2017 val2017)

| Metric | Score |
|--------|------:|
| BLEU-1 | _TBD_ |
| BLEU-2 | _TBD_ |
| BLEU-3 | _TBD_ |
| BLEU-4 | _TBD_ |
| CIDEr  | _TBD_ |
| ROUGE-L| _TBD_ |
| METEOR | _TBD_ |

## Quickstart (local)

```bash
# Install torch (CPU or CUDA — platform specific)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install the package + dev deps
pip install -e .[dev]
```

## Reproducing on Kaggle

**Step 1 — Cache features** (one-time, ~1h on P100):
Open `notebooks/cache_features_kaggle.ipynb` → add `awsaf49/coco-2017-dataset` → run all cells → upload `/kaggle/working/features/` as a Kaggle Dataset.

**Step 2 — Train** (`notebooks/train_kaggle.ipynb`):
Add COCO 2017 + your cached features Dataset + HF Hub token as Kaggle Secret → run all cells. Auto-resumes from `latest.pt`.

## Project layout

```
src/
├── data/          # dataset (COCO 2017), vocab, transforms
├── models/        # encoder (ResNet50), decoder (LSTM)
├── train.py       # hand-rolled training loop
├── evaluate.py    # pycocoevalcap BLEU/CIDEr/ROUGE/METEOR
├── inference.py   # beam_search + greedy_decode
└── captioner.py   # Captioner.from_pretrained()
```

## Design decisions

- **COCO 2017 annotations directly** (not Karpathy split). Simpler setup, no COCO 2014/2017 path mismatch. Numbers on the validation set.
- **Cached encoder features.** Running frozen ResNet50 once and caching 2048-d fp16 vectors drops epoch time from ~3h to ~20min.
- **torch is optional.** Kaggle pre-installs it; `pyproject.toml` lists torch under `[project.optional-dependencies] torch`. Local install via `pip install torch`.
- **Augmentation disabled in v1.** Cached feature strategy makes per-epoch augmentation impossible without re-encoding.
- **Hand-rolled training loop.** No Lightning, no HF Trainer. Fully transparent.

## Citing
- **Show, Attend and Tell** — Xu et al., 2015.
- **Microsoft COCO** — Lin et al., 2014.

## License
[MIT](LICENSE)