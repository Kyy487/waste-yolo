from pathlib import Path

import numpy as np
import pytest

from src.api.inference import (
    ONNXInference,
    preprocess_frame,
    probabilities_from_output,
    top_k_from_probabilities,
)


def test_preprocess_frame_shape_and_range() -> None:
    """Preprocessing returns NCHW float input in [0, 1]."""
    frame = np.full((48, 96, 3), 128, dtype=np.uint8)
    tensor = preprocess_frame(frame, 224)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == np.float32
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0


def test_preprocess_frame_keeps_aspect_ratio_with_padding() -> None:
    """Non-square frames are padded instead of stretched."""
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[:, :100] = 255
    tensor = preprocess_frame(frame, 224, padding_color=0)[0]
    assert tensor.shape == (3, 224, 224)
    assert float(tensor[:, 0, :].max()) == 0.0
    assert float(tensor[:, -1, :].max()) == 0.0


def test_preprocess_frame_rejects_invalid_input() -> None:
    """Empty or malformed frames are rejected."""
    with pytest.raises(ValueError):
        preprocess_frame(np.zeros((0, 0, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        preprocess_frame(np.zeros((10, 10), dtype=np.uint8))


def test_probabilities_from_logits_are_normalized() -> None:
    """Raw logits are converted with softmax."""
    probabilities = probabilities_from_output([2.0, 1.0, 0.1], 3)
    assert probabilities.shape == (3,)
    assert float(probabilities.sum()) == pytest.approx(1.0)
    assert int(np.argmax(probabilities)) == 0


def test_probabilities_from_output_accepts_probabilities() -> None:
    """Outputs that already sum to one are kept as-is."""
    values = np.array([0.7, 0.2, 0.1], dtype=np.float32)
    result = probabilities_from_output(values, 3)
    assert np.allclose(result, values)


def test_probabilities_from_output_rejects_wrong_size() -> None:
    """Output size must match the number of classes."""
    with pytest.raises(ValueError):
        probabilities_from_output([1.0, 2.0], 3)


def test_top_k_is_sorted_and_limited() -> None:
    """Top-k scores are sorted descending and limited by class count."""
    probabilities = np.array([0.1, 0.5, 0.4], dtype=np.float32)
    result = top_k_from_probabilities(probabilities, ["a", "b", "c"], 2)
    assert result["predicted_class"] == "b"
    assert [score["class"] for score in result["top_k"]] == ["b", "c"]
    assert result["top_k"][0]["confidence"] >= result["top_k"][1]["confidence"]


def test_onnx_load_requires_model_file(tmp_path: Path) -> None:
    """Loading fails clearly when the ONNX file is missing."""
    config = {
        "model": {
            "path": "missing.onnx",
            "card": "model_card.json",
            "classes_source": "data.yaml",
            "input_size": 224,
        },
        "inference": {},
    }
    with pytest.raises(FileNotFoundError):
        ONNXInference.load(config, tmp_path)
