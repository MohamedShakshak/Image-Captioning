from config import Config


def test_yaml_load_and_override():
    import tempfile

    import yaml

    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/cfg.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(
                {
                    "seed": 42,
                    "train": {"epochs": 20, "lr": 4.0e-4},
                    "model": {"hidden_dim": 512},
                },
                f,
            )
        cfg = Config.from_yaml(path)
        assert cfg.seed == 42
        assert cfg.train["epochs"] == 20

        cfg.override("train.epochs", "30")
        assert cfg.train["epochs"] == 30

        cfg.override("train.lr", "0.001")
        assert cfg.train["lr"] == 0.001

        cfg.override("train.pin_memory", "true")
        assert cfg.train["pin_memory"] is True
