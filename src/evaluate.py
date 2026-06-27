"""Evaluation: per-epoch val-loss (used during training) + final pycocoevalcap.

See PLAN.md § Eval."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import Config
from data.dataset import CachedCOCODataset, collate_fn
from data.vocab import Vocab
from inference import beam_search
from models.decoder import Decoder
from train import load_karpathy_split


def evaluate(cfg: Config) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocab.load(cfg.data["vocab_path"])
    karpathy = load_karpathy_split(cfg.data["karpathy_split"])

    test_ds = CachedCOCODataset(cfg.data["features_dir"], karpathy, vocab, split="test")
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.eval["batch_size"],
        shuffle=False,
        num_workers=cfg.train["num_workers"],
        pin_memory=cfg.train["pin_memory"],
        collate_fn=collate_fn,
    )

    decoder = Decoder(
        vocab_size=vocab.size,
        hidden_dim=cfg.model["hidden_dim"],
        embedding_dim=cfg.model["embedding_dim"],
        dropout=cfg.model["dropout"],
    ).to(device)
    ckpt = torch.load(Path(cfg.checkpoint["dir"]) / "best.pt", map_location="cpu")
    decoder.load_state_dict(ckpt["model"])
    decoder.eval()

    preds: dict[str, list[dict]] = {}
    gts: dict[str, list[dict]] = {}
    img_id = 0
    with torch.no_grad():
        for features, _captions, _l in test_loader:
            features = features.to(device)
            for i in range(features.size(0)):
                feats = features[i]
                results = beam_search(
                    decoder,
                    feats.unsqueeze(0),
                    beam_size=cfg.eval["beam_size"],
                    length_norm_alpha=cfg.eval["length_norm_alpha"],
                )
                tokens, _score = results[0]
                text = " ".join(vocab.decode(tokens))
                preds[f"img_{img_id}"] = [{"caption": text}]
                # Ground truth from Karpathy
                entry = test_ds.entries[img_id]
                gts[f"img_{img_id}"] = [{"caption": s["raw"]} for s in entry["sentences"]]
                img_id += 1

    out_dir = Path("eval_out")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "preds.json").write_text(json.dumps(preds, indent=2))
    (out_dir / "gts.json").write_text(json.dumps(gts, indent=2))

    # BLEU is pure-python in pycocoevalcap; CIDEr/METEOR/SPICE need Java.
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
        meteor = Meteor()
        m_score, _ = meteor.compute_score(gts, preds)
        print(f"METEOR: {m_score:.4f}")
    else:
        print("(full_metrics=false — CIDEr/METEOR skipped; run on Kaggle for full suite)")


if __name__ == "__main__":
    from config import parse_args

    cfg, _ = parse_args()
    evaluate(cfg)
