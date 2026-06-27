# Image Captioning — PyTorch v1 Implementation Plan

## Goal
Portfolio/CV project. Clean, runnable, impressive-at-a-glance PyTorch port of an old Keras ResNet50 + LSTM image captioner. Vanilla LSTM v1 (attention deferred as documented v2 milestone). Trained on COCO via Kaggle free P100. Deployed as Streamlit demo on HF Spaces.

## Target metrics (literature-comparable)
- BLEU-4 ≈ 0.27–0.29 on COCO Karpathy test split
- Full pycocoevalcap suite (BLEU-1..4, CIDEr, ROUGE-L, METEOR) evaluated on Kaggle (Java available)
- BLEU-1..4 only available locally (pure-Python subset)

## Architecture

### Encoder
- `torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)`, frozen, FC stripped.
- Output: pooled 2048-d vector per image.
- Cached to disk pre-training as fp16 `.npy` (~1GB for COCO full). Encoder never re-run during training.
- `models/encoder.py` wraps this; `scripts/cache_features.py` does the one-time pass.

### Decoder
- `nn.Linear(2048 -> 512)` ×2 for `h0` / `c0` init projections (`tanh`).
- `nn.Embedding(vocab_size, 512)`.
- `nn.LSTM(512, 512, num_layers=1, dropout=0.5)`.
- `nn.Linear(512, vocab_size)`.
- Inputs: `<start> + tokens + <end>` per caption.
- Loss: offset-aligned (don't predict `<start>` as a target).
- Init coupling: `h0 = tanh(init_h(features))`, `c0 = tanh(init_c(features))`.

### Vocabulary
- Custom build (`scripts/build_vocab.py`), saves `vocab.json`.
- `min_freq=5`, specials ordered: `<pad>=0, <start>=1, <end>=2, <unk>=3`.
- ~10k words after preprocessing.
- Caption preprocessing: lowercase, strip punctuation, PTB-style contraction split (built-in dict of ~20 contractions, no dep).
- `<start>` / `<end>` injected at `collate_fn` time, NOT during vocab build.
- `ignore_index=<pad>=0` in loss.

### Inference masking
- `<unk>` and `<pad>` logits → `-inf` before beam expansion. Guarantees no `<unk>` / `<pad>` in demo output.

## Config (`configs/default.yaml`)
```yaml
seed: 42
data:
  coco_root: /kaggle/input/coco-2017
  karpathy_split: /kaggle/input/coco-karpathy/dataset_coco.json
  features_dir: /kaggle/input/coco-features
  vocab_path: vocab.json
  image_size: 224
  resize: 256
model:
  hidden_dim: 512
  embedding_dim: 512
  dropout: 0.5
  vocab_min_freq: 5
train:
  batch_size: 32
  epochs: 20
  lr: 4.0e-4
  weight_decay: 1.0e-5
  label_smoothing: 0.1
  grad_clip: 1.0
  num_workers: 2
  pin_memory: true
  amp: false   # P100 has no tensor cores; LSTM fp16 NaN risk
optim:
  scheduler: reduce_on_plateau
  factor: 0.5
  patience: 2
eval:
  beam_size: 3
  length_norm_alpha: 0.7
  batch_size: 16
  full_metrics: false   # set true on Kaggle (Java available)
checkpoint:
  dir: /kaggle/working/checkpoints
  save_latest: true
  save_best: true
hf:
  repo: MohamedShakshak/image-captioning-pytorch
  token_env: HF_TOKEN
misc:
  log_every: 50
```

Loaded into `Config` dataclass via `src/image_captioning/config.py`; argparse overrides (`-- Train.epochs 30` etc.).

## File Tree (`src/` layout)
```
Image-Captioning/
├── README.md
├── PLAN.md
├── pyproject.toml                 # uv, ruff, mypy, pytest, package metadata (src layout)
├── requirements.txt               # Kaggle fallback, exact-pinned
├── LICENSE                        # MIT
├── Dockerfile                    # python:3.11-slim, CPU torch, package, HF Hub pull at startup
├── .gitignore
├── .github/workflows/ci.yml       # ruff + mypy + pytest via uv
├── configs/default.yaml
├── notebooks/train_kaggle.ipynb  # thin launcher, outputs stripped via nbstripout
├── src/image_captioning/
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py            # COCODataset + collate_fn (pad+lengths)
│   │   ├── vocab.py              # Vocab class, build/save/load (json)
│   │   └── transforms.py         # train (RandomCrop+flip, documented as disabled-v1), inference (CenterCrop)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoder.py            # ResNet50 wrapper
│   │   └── decoder.py            # LSTM + init projections
│   ├── train.py                  # hand-rolled loop, AMP-free, checkpoint resume
│   ├── evaluate.py               # val-loss per-epoch + final pycocoevalcap on Karpathy test
│   ├── inference.py              # beam_search(), greedy_decode() functions (CPU-testable)
│   └── captioner.py              # Captioner class, from_pretrained(hf_repo), caption() -> CaptionResult
├── scripts/
│   ├── download_coco.py          # --split {tiny,mini,full}; fetches Karpathy json
│   ├── cache_features.py         # one-time encoder pass, inference transforms only, fp16 .npy
│   ├── build_vocab.py            # one-time, saves vocab.json
│   └── plot_curves.py            # matplotlib curve PNG from metrics.json
├── app/streamlit_app.py          # Captioner cached @st.cache_resource, image upload + caption + beam panel
└── tests/
    ├── test_vocab.py
    ├── test_collate.py
    ├── test_encoder.py
    ├── test_decoder.py
    ├── test_beam.py              # determinism + <end> + length-norm
    ├── test_checkpoint.py        # save/load round-trip
    └── test_config.py            # YAML load + argparse override
```

## Training Pipeline (Kaggle)

1. `notebooks/train_kaggle.ipynb` clones the repo, `pip install -e .`, mounts:
   - `coco-2017` Kaggle Dataset (images + annotations)
   - `coco-karpathy` Kaggle Dataset (dataset_coco.json)
   - `coco-features` Kaggle Dataset (cached encoder features, precomputed by `cache_features.py`)
2. Pre-step (one-time, separate session): run `scripts/cache_features.py` + `scripts/build_vocab.py`. Push outputs to a Kaggle Dataset for reuse.
3. Runs `python -m image_captioning.train --config configs/default.yaml --resume`. Resume auto-finds latest checkpoint.
4. Per-iteration: `tqdm` bar + per-batch loss appended to in-memory list.
5. Per-epoch: write `metrics.json` row `{epoch, train_loss, val_loss, lr, timestamp}`. Save `latest.pt` (overwrite) and `best.pt` (overwrite only on new-best val loss).
6. On new-best or on training complete: `huggingface_hub` push `best.pt` to HF Hub repo `MohamedShakshak/image-captioning-pytorch`.
7. Final block in notebook: `python -m image_captioning.evaluate --full_metrics true`. Reports BLEU-1..4 + CIDEr + ROUGE-L + METEOR on Karpathy test (~5k images), batched beam, ~2 min.

## Eval

- **Per-epoch**: val-loss only (cheap, drives `best.pt` selection).
- **End-of-training**: `evaluate.py` runs beam=3 + length-norm alpha=0.7 + unk/pad masked, batched across 16 images at a time. Metrics via `pycocoevalcap` on Kaggle, BLEU-1..4 only locally.

## Image Preprocessing
- **Training transforms** (`data/transforms.py`): `Resize(256) → RandomCrop(224) → RandomHorizontalFlip → ToTensor → Normalize(ImageNet)`.
  - **Documented as DISABLED for v1**: cached encoder features use `inference_transform`; running aug at train-time would require re-encoding per epoch, defeating the cache strategy. Documented tradeoff in README design-decisions.
- **Inference transforms** (used by cache_features.py, evaluate.py, Streamlit app): `Resize(256) → CenterCrop(224) → ToTensor → Normalize(ImageNet)`.

## Packaging
- `uv` + `pyproject.toml` (`packages.find` `where=["src"]`).
- Loose pins in `pyproject.toml` (`torch>=2.0`), exact-pinned `requirements.txt` for Kaggle + CI.
- CI: `uv pip install -e .[dev]` → `ruff check` → `ruff format --check` → `mypy src/` → `pytest -q`. CPU-only, synthetic data, <2 min.

## Demo
- `Dockerfile`: `python:3.11-slim` + `streamlit` + CPU `torch` + package local install.
- HF Spaces deployment via Dockerfile runtime.
- `Captioner.from_pretrained(hf_repo)` on container startup (~10s cold start, downloads ~250MB weights from HF Hub).
- UI: image upload → `Captioner.caption()` → top caption + beam panel (3 beams with cumulative log-probs).

## README (level C)
- Badges: CI status, HF Spaces link, license (MIT), Python version.
- Mermaid architecture diagram (encoder → cached features → decoder → beam → evaluator).
- Results table: final BLEU-1..4 / CIDEr / ROUGE-L / METEOR on Karpathy test split.
- Design decisions section:
  - Vanilla-LSTM-first with attention as documented v2 milestone.
  - Cache-features fp16 trick (5x epoch speedup).
  - Karpathy split for literature-comparable numbers.
  - Aug disabled for cache efficiency in v1.
- Install: `uv sync`, `download_coco.py --split tiny`, `pytest -q`.
- Links: HF Spaces demo, HF Hub model, Kaggle notebook artifact.
- Cites: Show, Attend and Tell (Xu et al. 2015), Karpathy split (Karpathy & Fei-Fei 2015), COCO (Lin et al. 2014).

## Release
- Tag `v1.0.0`. GitHub Release attach:
  - `train_kaggle_outputs.ipynb` (full-output version of the Kaggle notebook)
  - link to HF Hub model repo
  - `metrics.json` + `curves.png`
  - Release notes: Keras-vs-PyTorch 3-row comparison table (BLEU-1..4) showing improvement.

## Git Workflow
- `main` + feature branches, squash-merge, Conventional Commit prefixes.
- Branches planned:
  - `chore/packaging` (pyproject, requirements, gitignore, LICENSE, PLAN.md)
  - `feat/config` (config.py + default.yaml)
  - `feat/data` (dataset, vocab, transforms, download_coco, cache_features, build_vocab)
  - `feat/models` (encoder, decoder)
  - `feat/train` (train.py, checkpoint logic, metrics logging)
  - `feat/eval` (evaluate.py, pycocoevalcap integration)
  - `feat/inference` (inference.py + captioner.py)
  - `feat/app` (streamlit_app.py + Dockerfile)
  - `feat/ci` (ci.yml + tests/)
  - `docs/readme`
- No `CONTRIBUTING.md`, no PR template (solo project).

## Loose ends (one-liner values)
- License: MIT
- Python: 3.11
- Seed: 42 (`random`, `numpy`, `torch`, CUDA deterministic)
- Cache file format: `.npy` fp16 (portable across torch versions)
- DataLoader: `num_workers=2`, `pin_memory=True`
- Eval batch size: 16
- HF Hub repo: `MohamedShakshak/image-captioning-pytorch`
- No gradient accumulation
- Weight init: PyTorch defaults (LSTM Xavier-uniform, Linear Kaiming, embedding ~N(0,1))