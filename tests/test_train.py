from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.metrics import f1_score

from src.train.train import (
    EXPERIMENT_COLUMNS,
    assert_new_run_id,
    macro_f1_from_confusion_matrix,
    resolve_weights,
    validation_metrics,
)


def test_macro_f1_matches_sklearn() -> None:
    """Macro-F1 from the confusion matrix matches explicit predictions."""
    matrix = np.array([[5, 1, 0], [0, 3, 1], [0, 2, 4]])
    y_true: list[int] = []
    y_pred: list[int] = []
    for predicted in range(matrix.shape[0]):
        for actual in range(matrix.shape[1]):
            y_true.extend([actual] * int(matrix[predicted, actual]))
            y_pred.extend([predicted] * int(matrix[predicted, actual]))
    expected = f1_score(y_true, y_pred, average="macro", zero_division=0)
    assert macro_f1_from_confusion_matrix(matrix) == pytest.approx(expected)


def test_empty_or_invalid_confusion_matrix() -> None:
    """Empty matrices give zero F1 and non-square matrices are rejected."""
    assert macro_f1_from_confusion_matrix(np.zeros((3, 3))) == 0.0
    with pytest.raises(ValueError):
        macro_f1_from_confusion_matrix(np.zeros((2, 3)))


def test_validation_metrics_uses_ultralytics_fields() -> None:
    """Validation metrics read top-1 accuracy and the confusion matrix."""
    metrics = SimpleNamespace(
        results_dict={"metrics/accuracy_top1": 0.75},
        confusion_matrix=SimpleNamespace(matrix=np.array([[3, 1], [0, 4]])),
    )
    values = validation_metrics(metrics)
    assert values["val_acc"] == 0.75
    assert 0.0 <= values["val_macro_f1"] <= 1.0
    assert validation_metrics(None) == {"val_acc": None, "val_macro_f1": None}


def test_duplicate_run_id_is_rejected(tmp_path: Path) -> None:
    """A run ID already present in the experiment table cannot be reused."""
    path = tmp_path / "experiments.csv"
    path.write_text(
        ",".join(EXPERIMENT_COLUMNS) + "\nE1_x,,0,0,0,,,,,\n", encoding="utf-8"
    )
    assert_new_run_id(path, "E2_x")
    with pytest.raises(ValueError):
        assert_new_run_id(path, "E1_x")


def test_resolve_weights_prefers_local_checkpoint(tmp_path: Path) -> None:
    """Local weights win over a model name that Ultralytics would download."""
    local = tmp_path / "local.pt"
    local.write_bytes(b"weights")
    assert resolve_weights("local.pt", tmp_path) == str(local)
    assert resolve_weights("yolov8n-cls.pt", tmp_path) == "yolov8n-cls.pt"
