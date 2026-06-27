"""Configuration dataclass + YAML loader + argparse overrides."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    """Top-level experiment config loaded from YAML, overridable via CLI dotted keys."""

    seed: int = 42
    data: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    train: dict[str, Any] = field(default_factory=dict)
    optim: dict[str, Any] = field(default_factory=dict)
    eval: dict[str, Any] = field(default_factory=dict)
    checkpoint: dict[str, Any] = field(default_factory=dict)
    hf: dict[str, Any] = field(default_factory=dict)
    misc: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw)

    def override(self, dotted_key: str, value: str) -> None:
        """Apply a dotted-key override like `train.epochs=30`."""
        parts = dotted_key.split(".")
        node: dict[str, Any] = self.__dict__[parts[0]]
        for p in parts[1:-1]:
            node = node[p]
        # Infer bool/int/float/str from the incoming string.
        v: bool | int | float | str
        if value.lower() in {"true", "false"}:
            v = value.lower() == "true"
        else:
            try:
                v = int(value)
            except ValueError:
                try:
                    v = float(value)
                except ValueError:
                    v = value
        node[parts[-1]] = v


def parse_args() -> tuple[Config, argparse.Namespace]:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config.")
    p.add_argument(
        "overrides",
        nargs="*",
        help="Dotted-key overrides, e.g. train.epochs=30 model.dropout=0.3",
    )
    args = p.parse_args()
    cfg = Config.from_yaml(args.config)
    for ov in args.overrides:
        ov = ov.removeprefix("--")  # accept both key=value and --key=value
        if "=" not in ov:
            raise SystemExit(f"Override must be key=value, got: {ov}")
        key, value = ov.split("=", 1)
        key = key.replace("-", "_")  # normalize dashes to underscores (argparse convention)
        cfg.override(key, value)
    return cfg, args
