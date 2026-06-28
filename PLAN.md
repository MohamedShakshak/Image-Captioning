# Image Captioning — Implementation Plan (v1, shipped)

## Goal
Portfolio/CV project. Clean, runnable PyTorch image captioner: ResNet50 encoder + LSTM decoder. Trained on COCO 2017 via Kaggle P100. Deployed as Streamlit demo on HF Spaces.

## Results (COCO 2017 val2017)
| Metric | Score |
|--------|------:|
| BLEU-1 | 0.5916 |
| BLEU-2 | 0.4077 |
| BLEU-3 | 0.2842 |
| BLEU-4 | 0.2034 |
| CIDEr  | 0.6035 |
| ROUGE-L| 0.4048 |
| METEOR | _TBD_ |

## Architecture

### Encoder
`torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)`, frozen, FC stripped, outputs pooled 2048-d vector. Cached to disk as fp16 `.npy` files (12-digit numeric IDs matching COCO 2017).

### Decoder
`nn.Linear(2048->512)×2` for `h0`/`c0` init projections (`tanh`), `nn.Embedding(vocab_size, 512)`, `nn.LSTM(512,512,num_layers=1,dropout=0.0)`, `nn.Linear(512, vocab_size)`. Inputs: `<start>+tokens+<end>`. Loss offset-aligned (don't predict `<start>`). Init coupling: `h0=tanh(init_h(features))`, `c0=tanh(init_c(features))`.

### Vocabulary
Custom vocab (`scripts/build_vocab.py`), saves `vocab.json`. `min_freq=5`, specials: `<pad>=0, <start>=1, <end>=2, <unk>=3`. ~10k words. Preprocessing: lowercase, strip punctuation, PTB-style contraction split. `<start>`/`<end>` injected at `collate_fn` time. `ignore_index=0` in loss. Inference masks `<unk>`/`<pad>` to `-inf`.

## Config (`configs/default.yaml`)
```yaml
seed: 42
data:
  coco_root: /kaggle/input/coco-2017
  annotation_train: /kaggle/input/coco-2017/annotations/captions_train2017.json
  annotation_val: /kaggle/input/coco-2017/annotations/captions_val2017.json
  features_dir: /kaggle/input/coco-features
  vocab_path: vocab.json
  image_size: 224
  resize: 256
model: {hidden_dim: 512, embedding_dim: 512, dropout: 0.5, vocab_min_freq: 5}
train: {batch_size: 32, epochs: 20, lr: 4e-4, weight_decay: 1e-5, label_smoothing: 0.1, grad_clip: 1.0, num_workers: 2, pin_memory: true, amp: false}
optim: {scheduler: reduce_on_plateau, factor: 0.5, patience: 2}
eval: {beam_size: 3, length_norm_alpha: 0.7, batch_size: 16, full_metrics: false}
checkpoint: {dir: /kaggle/working/checkpoints, save_latest: true, save_best: true}
hf: {repo: MohamedShakshak/image-captioning-pytorch, token_env: HF_TOKEN}
misc: {log_every: 50}
```
Overrides use `key=value` syntax (or `--key=value`). Dashes normalize to underscores.

## File Tree
```
Image-Captioning/
├── README.md, PLAN.md, LICENSE
├── pyproject.toml        # setuptools, src/ layout, ruff/mypy/pytest config
├── requirements.txt      # pinned non-torch deps (Kaggle fallback)
├── Dockerfile            # python:3.11-slim + CPU torch + Streamlit
├── .github/workflows/ci.yml  # ruff -> mypy -> pytest
├── configs/default.yaml
├── notebooks/
│   ├── cache_features_kaggle.ipynb  # one-time: cache encoder features + build vocab
│   └── train_kaggle.ipynb           # training launcher
├── src/                            # flat layout, no sub-package nesting
│   ├── __init__.py                 # __version__
│   ├── config.py                   # Config dataclass + YAML + argparse overrides
│   ├── data/                       # CachedCOCODataset, collate_fn, Vocab, transforms
│   ├── models/                     # Encoder (ResNet50), Decoder (LSTM)
│   ├── train.py                    # hand-rolled loop, checkpoint resume, HF push
│   ├── evaluate.py                 # pycocoevalcap (BLEU/CIDEr/ROUGE/METEOR)
│   ├── inference.py                # greedy_decode + beam_search (CPU-testable)
│   └── captioner.py                # Captioner.from_pretrained()
├── scripts/
│   ├── cache_features.py           # one-time: runs ResNet50 over all COCO 2017 images
│   ├── build_vocab.py              # builds vocab.json from COCO 2017 captions
│   ├── download_coco.py            # placeholders
│   └── plot_curves.py              # matplotlib curves from metrics.json
├── app/streamlit_app.py            # upload image -> caption + beam panel
└── tests/                          # 7 files, 11 tests, CPU synthetic data
```

## Training Pipeline (Kaggle)

### Step 1: `cache_features_kaggle.ipynb` (one-time, ~1h)
- Install torch 2.1.2 CUDA 11.8 (P100 sm_60 compatibility)
- Load COCO 2017 annotations (`captions_train2017.json` + `captions_val2017.json`)
- Run frozen ResNet50 over all ~123k images
- Save fp16 `.npy` files as `000000XXXXXX.npy` (12-digit numeric IDs)
- Build `vocab.json` from all captions
- Upload `/kaggle/working/features/` as a Kaggle Dataset

### Step 2: `train_kaggle.ipynb` (per training run)
- Clone repo, `pip install -e .` (does NOT touch torch)
- Mount COCO 2017 + cached features Dataset
- Build Config from YAML + overrides (annotation paths, feature dir, etc.)
- `train()`: hand-rolled loop, auto-resume from `latest.pt`
- Per-epoch: `tqdm` + batch loss → metrics.json. Save `latest.pt` + `best.pt` (val-loss driven)
- On new-best: push `best.pt` + `vocab.json` to HF Hub
- Final cell: `evaluate.py` runs pycocoevalcap on COCO val2017

## Image Preprocessing
- Inference transform (used by cache_features, eval, demo): `Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize(ImageNet)`
- Training transform (DISABLED in v1 due to feature caching): `RandomCrop + RandomHorizontalFlip`

## Packaging
- setuptools with `package-dir = {"": "src"}`. Editable install adds `src/` to path.
- `torch`/`torchvision` in `[project.optional-dependencies] torch` (platform-specific; Kaggle pre-installs).
- `requirements.txt` pinned (no torch). CI: `pip install -e .[dev]` -> ruff -> mypy -> pytest.
- P100 compatibility: use torch 2.1.x CUDA 11.8. torch 2.2+ CUDA 12.x drops sm_60 support.

## Demo
- `Dockerfile`: `python:3.11-slim` + CPU torch + Streamlit + package
- HF Spaces deployment via Dockerfile runtime
- `Captioner.from_pretrained(hf_repo)` downloads ~250MB weights at startup (~10s cold start)
- UI: image upload → encoder → beam search k=3 → top caption + beam panel

## Git Workflow
- `main` + feature branches, squash-merge, Conventional Commits
- CI runs on push/PR: lint + typecheck + pytest

## Loose ends
- MIT license, Python 3.11, seed 42
- Cache file format: `.npy` fp16
- No gradient accumulation, PyTorch default weight init
- P100-specific: torch 2.1.2 CUDA 11.8, no AMP (no tensor cores, LSTM+fp16 NaN risk)