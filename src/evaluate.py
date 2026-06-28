"""Evaluation: per-epoch val-loss (used during training) + final pycocoevalcap.

Uses COCO 2017 val2017 as the evaluation set.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from config import Config
from data.dataset import CachedCOCODataset
from data.vocab import Vocab
from inference import beam_search
from models.decoder import Decoder


def _build_ground_truth(annotation_json: str | Path) -> dict[str, list[str]]:
    """Build pycocoevalcap-format ground truth: {img_id: [caption_str, ...]}"""
    with open(annotation_json) as f:
        data = json.load(f)

    gts: dict[int, list[str]] = {}
    for ann in data["annotations"]:
        iid = ann["image_id"]
        if iid not in gts:
            gts[iid] = []
        gts[iid].append(ann["caption"])

    result: dict[str, list[str]] = {}
    for iid, caps in gts.items():
        key = f"{iid:012d}"
        result[key] = caps
    return result


def evaluate(cfg: Config) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocab.load(cfg.data["vocab_path"])

    test_ds = CachedCOCODataset(cfg.data["features_dir"], cfg.data["annotation_val"], vocab)

    decoder = Decoder(
        vocab_size=vocab.size,
        hidden_dim=cfg.model["hidden_dim"],
        embedding_dim=cfg.model["embedding_dim"],
        dropout=cfg.model["dropout"],
    ).to(device)
    ckpt = torch.load(Path(cfg.checkpoint["dir"]) / "best.pt", map_location="cpu")
    decoder.load_state_dict(ckpt["model"])
    decoder.eval()

    gts = _build_ground_truth(cfg.data["annotation_val"])
    preds: dict[str, list[str]] = {}

    decoder.eval()
    with torch.no_grad():
        for idx in range(len(test_ds)):
            img_id, features, _ = test_ds[idx]
            features = features.to(device).unsqueeze(0)
            results = beam_search(
                decoder,
                features,
                beam_size=cfg.eval["beam_size"],
                length_norm_alpha=cfg.eval["length_norm_alpha"],
            )
            tokens, _score = results[0]
            text = " ".join(vocab.decode(tokens))
            key = f"{img_id:012d}"
            preds[key] = [text]

    out_dir = Path("eval_out")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "preds.json").write_text(json.dumps(preds, indent=2))
    (out_dir / "gts.json").write_text(json.dumps(gts, indent=2))

    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge

    bleu = Bleu(4)
    score, _ = bleu.compute_score(gts, preds)
    names = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]
    for n, s in zip(names, score, strict=False):
        print(f"{n}: {s:.4f}")

    rouge = Rouge()
    r_score, _ = rouge.compute_score(gts, preds)
    print(f"ROUGE-L: {r_score:.4f}")

    if cfg.eval["full_metrics"]:
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.meteor.meteor import Meteor

        cider = Cider()
        c_score, _ = cider.compute_score(gts, preds)
        print(f"CIDEr: {c_score:.4f}")
        try:
            meteor = Meteor()
            m_score, _ = meteor.compute_score(gts, preds)
            print(f"METEOR: {m_score:.4f}")
        except Exception as e:
            print(f"METEOR skipped (Java subprocess issue): {e}")
    else:
        print("(full_metrics=false — CIDEr/METEOR skipped; run on Kaggle for full suite)")


if __name__ == "__main__":
    from config import parse_args

    cfg, _ = parse_args()
    evaluate(cfg)
