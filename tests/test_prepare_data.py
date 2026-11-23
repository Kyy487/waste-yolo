from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.data.prepare_data import build_dataset, preprocess_image
from src.utils.io import read_yaml

ROOT = Path(__file__).resolve().parents[1]


def preprocess_config() -> dict:
    return {
        "image_size": 224,
        "keep_aspect_ratio": True,
        "clahe": False,
        "resize_interpolation": "INTER_AREA",
        "padding_color": 0,
        "jpeg_quality": 95,
    }


def test_preprocess_pads_and_converts_bgr_to_rgb() -> None:
    """A 200x100 BGR image becomes a padded 224x224 RGB image."""
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    output = preprocess_image(image, preprocess_config())

    assert output.shape == (224, 224, 3)
    assert output[0, 0].tolist() == [0, 0, 0]
    assert output[56, 112].tolist() == [0, 0, 255]


def test_zero_image_size_raises_value_error() -> None:
    """Invalid target size is rejected."""
    config = preprocess_config()
    config["image_size"] = 0
    with pytest.raises(ValueError):
        preprocess_image(np.zeros((8, 8, 3), dtype=np.uint8), config)


def test_build_dataset_end_to_end(tmp_path: Path) -> None:
    """The pipeline writes splits, removals, and preprocessed class folders."""
    config = deepcopy(read_yaml(ROOT / "configs" / "data.yaml"))
    config["classes"] = ["glass", "metal"]
    for class_index, class_name in enumerate(config["classes"]):
        class_dir = tmp_path / "data" / "raw" / class_name
        class_dir.mkdir(parents=True)
        for image_index in range(6):
            rng = np.random.default_rng(class_index * 100 + image_index)
            image = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
            cv2.imwrite(str(class_dir / f"{class_name}_{image_index}.jpg"), image)
        duplicate = class_dir / "duplicate.jpg"
        duplicate.write_bytes((class_dir / f"{class_name}_0.jpg").read_bytes())

    counts = build_dataset(config, tmp_path)

    assert sum(counts.values()) == 12
    split_lines = (
        (tmp_path / "data" / "processed" / "splits.csv")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert split_lines[0] == "filepath,class,split,sha1"
    assert len(split_lines) == 13

    removed_lines = (
        (tmp_path / "data" / "processed" / "dedup_removed.csv")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert any("DUPLICATE" in line for line in removed_lines)

    audit_lines = (
        (tmp_path / "results" / "data_audit.csv")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert audit_lines[0] == "class,raw,final,removed"
    assert len(audit_lines) == 3
    assert (tmp_path / "results" / "class_distribution.png").is_file()

    written = list((tmp_path / "data" / "processed" / "dataset").rglob("*.jpg"))
    assert len(written) == 12
    assert all(
        image.shape == (224, 224, 3) for image in (cv2.imread(str(p)) for p in written)
    )
