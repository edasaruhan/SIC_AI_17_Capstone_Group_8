from pathlib import Path

import yaml


def test_sprint_zero_structure() -> None:
    required = [
        "configs/base.yaml",
        "data/raw",
        "data/interim",
        "data/processed",
        "docs/references/fikir-onerisi.pdf",
        "notebooks",
        "reports",
        "src",
    ]

    assert all(Path(path).exists() for path in required)


def test_seed_is_fixed() -> None:
    config = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))

    assert config["seed"] == 42
