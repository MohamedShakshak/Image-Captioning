"""Hand-rolled training loop. No Lightning, no AMP. Checkpoint resume supported.

See PLAN.md § Training Pipeline."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from data.dataset import CachedCOCODataset, collate_fn
from data.vocab import PAD_IDX, Vocab
from models.decoder import Decoder


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_ce_loss(
    logits: torch.Tensor,  # (B, T, V)
    targets: torch.Tensor,  # (B, T)
    label_smoothing: float = 0.1,
) -> torch.Tensor:
    """Masked cross-entropy with label smoothing. Targets shifted by 1 (don't predict <start>).

    Input logits at position t come from token t; we predict the *next* token.
    So we use logits[:, :-1] to predict targets[:, 1:].
    """
    # Shift: predict next token
    pred_logits = logits[:, :-1, :].contiguous()  # (B, T-1, V)
    gold = targets[:, 1:].contiguous()  # (B, T-1)

    loss = nn.functional.cross_entropy(
        pred_logits.reshape(-1, pred_logits.size(-1)),
        gold.reshape(-1),
        ignore_index=PAD_IDX,
        label_smoothing=label_smoothing,
        reduction="mean",
    )
    return loss


def save_checkpoint(
    path: str | Path,
    decoder: Decoder,
    optimizer: Adam,
    scheduler,
    epoch: int,
    best_val_loss: float,
) -> None:
    torch.save(
        {
            "model": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
        },
        path,
    )


def load_checkpoint(
    path: str | Path, decoder: Decoder, optimizer: Adam, scheduler
) -> tuple[int, float]:
    ckpt = torch.load(path, map_location="cpu")
    decoder.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return int(ckpt["epoch"]), float(ckpt["best_val_loss"])


def _latest_checkpoint(ckpt_dir: Path) -> Path | None:
    latest = ckpt_dir / "latest.pt"
    return latest if latest.exists() else None


def train(cfg: Config) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = Vocab.load(cfg.data["vocab_path"])

    train_ds = CachedCOCODataset(cfg.data["features_dir"], cfg.data["annotation_train"], vocab)
    val_ds = CachedCOCODataset(cfg.data["features_dir"], cfg.data["annotation_val"], vocab)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train["batch_size"],
        shuffle=True,
        num_workers=cfg.train["num_workers"],
        pin_memory=cfg.train["pin_memory"],
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train["batch_size"],
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

    optimizer = Adam(
        decoder.parameters(), lr=cfg.train["lr"], weight_decay=cfg.train["weight_decay"]
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=cfg.optim["factor"],
        patience=cfg.optim["patience"],
        mode="min",
    )

    ckpt_dir = Path(cfg.checkpoint["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    start_epoch, best_val_loss = 0, math.inf
    latest = _latest_checkpoint(ckpt_dir)
    if latest is not None:
        start_epoch, best_val_loss = load_checkpoint(latest, decoder, optimizer, scheduler)
        start_epoch += 1
        print(f"Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    metrics_log: list[dict] = []
    metrics_path = Path("metrics.json")
    if metrics_path.exists():
        metrics_log = json.loads(metrics_path.read_text())

    for epoch in range(start_epoch, cfg.train["epochs"]):
        decoder.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{cfg.train['epochs'] - 1}")
        for features, captions, _lengths in pbar:
            features = features.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = decoder(features, captions)
            loss = masked_ce_loss(logits, captions, cfg.train["label_smoothing"])
            loss.backward()
            nn.utils.clip_grad_norm_(decoder.parameters(), cfg.train["grad_clip"])
            optimizer.step()
            running += float(loss.item())
            if len(pbar) % cfg.misc["log_every"] == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss = running / max(1, len(train_loader))

        # Validation loss
        decoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, captions, _l in val_loader:
                features = features.to(device, non_blocking=True)
                captions = captions.to(device, non_blocking=True)
                logits = decoder(features, captions)
                val_loss += float(
                    masked_ce_loss(logits, captions, cfg.train["label_smoothing"]).item()
                )
        val_loss /= max(1, len(val_loader))

        scheduler.step(val_loss)
        cur_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} lr={cur_lr:.2e}"
        )

        metrics_log.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": cur_lr,
            }
        )
        metrics_path.write_text(json.dumps(metrics_log, indent=2))

        # Checkpoint
        if cfg.checkpoint["save_latest"]:
            save_checkpoint(
                ckpt_dir / "latest.pt", decoder, optimizer, scheduler, epoch, best_val_loss
            )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if cfg.checkpoint["save_best"]:
                save_checkpoint(
                    ckpt_dir / "best.pt", decoder, optimizer, scheduler, epoch, best_val_loss
                )
                _maybe_push_to_hf(ckpt_dir / "best.pt", cfg)

    print("Training complete.")


def _maybe_push_to_hf(best_pt_path: Path, cfg: Config) -> None:
    import os

    token = os.environ.get(cfg.hf["token_env"])
    if not token:
        return
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=str(best_pt_path),
            path_in_repo="best.pt",
            repo_id=cfg.hf["repo"],
            repo_type="model",
        )
        print(f"Pushed best.pt to HF Hub: {cfg.hf['repo']}")
    except Exception as e:
        print(f"HF upload skipped: {e}")


if __name__ == "__main__":
    from config import parse_args

    cfg, _ = parse_args()
    train(cfg)
