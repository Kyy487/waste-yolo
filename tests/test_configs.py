from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CLASSES = [
    "battery",
    "cardboard",
    "glass",
    "metal",
    "organic",
    "paper",
    "plastic",
    "textile",
    "trash",
]


def load_config(name: str) -> dict:
    """Load a YAML configuration from the configs directory."""
    return yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))


def test_data_config_contract() -> None:
    """Data config keeps the locked seed, class order, and split ratios."""
    config = load_config("data.yaml")
    assert config["seed"] == 42
    assert config["classes"] == CLASSES
    assert sum(config["split"].values()) == pytest.approx(1.0)


def test_train_and_api_configs_agree_on_image_size() -> None:
    """Training, API, and preprocessing use the same input resolution."""
    train = load_config("train.yaml")
    api = load_config("api.yaml")
    data = load_config("data.yaml")
    assert train["training"]["imgsz"] == 224
    assert api["model"]["input_size"] == train["training"]["imgsz"]
    assert data["preprocess"]["image_size"] == train["training"]["imgsz"]


def test_missing_config_raises_file_not_found() -> None:
    """A missing configuration file fails explicitly."""
    with pytest.raises(FileNotFoundError):
        (ROOT / "configs" / "missing.yaml").read_text(encoding="utf-8")
