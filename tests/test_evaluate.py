import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from sklearn.metrics import f1_score

from src.eval import evaluate as eval_module
from src.eval.evaluate import (
    ERROR_COLUMNS,
    PER_CLASS_COLUMNS,
    PredictionRecord,
    collect_eval_samples,
    compute_classification_metrics,
    evaluate_model,
    export_to_onnx,
    extract_error_rows,
    resolve_split_dir,
    validate_model_classes,
)

CLASSES = ["battery", "cardboard", "glass"]


class FakeYOLO:
    """Lightweight stand-in for Ultralytics YOLO used in unit tests."""

    def __init__(
        self, probs_by_name: dict[str, list[float]], names: dict[int, str]
    ) -> None:
        self._probs_by_name = probs_by_name
        self.names = names

    def predict(
        self,
        source: list[str],
        imgsz: int,
        batch: int,
        device: str | None = None,
        verbose: bool = False,
    ) -> list[Any]:
        results = []
        for item in source:
            filename = Path(item).name
            vector = np.asarray(self._probs_by_name[filename], dtype=float)
            results.append(SimpleNamespace(probs=SimpleNamespace(data=vector)))
        return results

    def export(self, format: str, imgsz: int) -> str:
        raise NotImplementedError


def test_compute_classification_metrics_matches_sklearn() -> None:
    """Metrics match sklearn macro-F1 and track worst class correctly."""
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 2, 2, 1]
    result = compute_classification_metrics(y_true, y_pred, CLASSES)

    expected_macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    assert result["num_samples"] == 6
    assert result["num_errors"] == 2
    assert result["accuracy"] == pytest.approx(4 / 6)
    assert result["macro_f1"] == pytest.approx(expected_macro_f1)
    assert result["min_class_f1"] == pytest.approx(0.5)
    assert result["worst_class"] in ("cardboard", "glass")
    assert result["confusion_matrix"].tolist() == [[2, 0, 0], [0, 1, 1], [0, 1, 1]]


def test_compute_classification_metrics_rejects_invalid_inputs() -> None:
    """Empty or mismatched label sequences raise ValueError."""
    with pytest.raises(ValueError):
        compute_classification_metrics([], [], CLASSES)
    with pytest.raises(ValueError):
        compute_classification_metrics([0, 1], [0], CLASSES)


def test_extract_error_rows_sorts_by_descending_confidence() -> None:
    """Only misclassified records are kept, highest confidence first."""
    records = [
        PredictionRecord("a.jpg", "battery", 0, "battery", 0, 0.99, 0.99),
        PredictionRecord("b.jpg", "cardboard", 1, "glass", 2, 0.65, 0.20),
        PredictionRecord("c.jpg", "glass", 2, "battery", 0, 0.91, 0.05),
    ]
    rows = extract_error_rows(records)
    assert [row["filepath"] for row in rows] == ["c.jpg", "b.jpg"]
    assert rows[0]["confidence"] == pytest.approx(0.91)
    assert rows[0]["true_confidence"] == pytest.approx(0.05)


def test_collect_eval_samples_round_robin_limit_and_errors(tmp_path: Path) -> None:
    """Round-robin sampling covers all classes; missing/empty dirs fail explicitly."""
    split_dir = tmp_path / "val"
    for cls in CLASSES:
        cls_dir = split_dir / cls
        cls_dir.mkdir(parents=True)
        for idx in range(2):
            (cls_dir / f"{cls}_{idx}.jpg").write_bytes(b"fake")

    samples = collect_eval_samples(split_dir, CLASSES, [".jpg"], limit=3)
    assert [sample.true_label for sample in samples] == CLASSES

    with pytest.raises(FileNotFoundError):
        collect_eval_samples(tmp_path / "missing", CLASSES, [".jpg"])
    with pytest.raises(ValueError):
        collect_eval_samples(split_dir, CLASSES, [".jpg"], limit=0)
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(ValueError):
        collect_eval_samples(empty_dir, CLASSES, [".jpg"])


def test_validate_model_classes_checks_exact_order() -> None:
    """Model class mapping must match configured alphabetical order."""
    validate_model_classes({0: "battery", 1: "cardboard", 2: "glass"}, CLASSES)
    with pytest.raises(ValueError):
        validate_model_classes({0: "glass", 1: "cardboard", 2: "battery"}, CLASSES)


def test_resolve_split_dir_rejects_unknown_split(tmp_path: Path) -> None:
    """Unsupported split names raise ValueError."""
    data_cfg = {
        "dataset": {
            "dataset_dir": "data/processed/dataset",
            "realworld_dir": "data/realworld",
        }
    }
    assert resolve_split_dir(data_cfg, tmp_path, "val") == (
        tmp_path / "data/processed/dataset/val"
    )
    assert resolve_split_dir(data_cfg, tmp_path, "realworld") == (
        tmp_path / "data/realworld"
    )
    with pytest.raises(ValueError):
        resolve_split_dir(data_cfg, tmp_path, "train")


def test_evaluate_model_writes_all_artifacts(tmp_path: Path) -> None:
    """End-to-end evaluation writes metrics.json, per-class CSV, CM, and errors.csv."""
    val_dir = tmp_path / "data" / "processed" / "dataset" / "val"
    for cls in CLASSES:
        (val_dir / cls).mkdir(parents=True)
    (val_dir / "battery" / "b1.jpg").write_bytes(b"img")
    (val_dir / "cardboard" / "c1.jpg").write_bytes(b"img")
    (val_dir / "glass" / "g1.jpg").write_bytes(b"img")

    weights = tmp_path / "models" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"weights")

    data_cfg = {
        "seed": 42,
        "dataset": {
            "dataset_dir": "data/processed/dataset",
            "realworld_dir": "data/realworld",
            "valid_extensions": [".jpg", ".png"],
        },
        "classes": CLASSES,
        "preprocess": {"image_size": 224},
    }
    api_cfg = {"model": {"path": "models/model.onnx", "input_size": 224}}

    fake_model = FakeYOLO(
        probs_by_name={
            "b1.jpg": [0.95, 0.03, 0.02],  # correct
            "c1.jpg": [0.10, 0.15, 0.75],  # wrong: predicted glass
            "g1.jpg": [0.05, 0.05, 0.90],  # correct
        },
        names={0: "battery", 1: "cardboard", 2: "glass"},
    )

    out_dir = tmp_path / "results" / "eval" / "val"
    summary = evaluate_model(
        weights_path=weights,
        data_config=data_cfg,
        api_config=api_cfg,
        root=tmp_path,
        split="val",
        output_dir=out_dir,
        model=fake_model,
    )

    assert summary["num_samples"] == 3
    assert summary["num_errors"] == 1
    assert summary["accuracy"] == pytest.approx(2 / 3)

    assert (out_dir / "metrics.json").is_file()
    assert (out_dir / "per_class_metrics.csv").is_file()
    assert (out_dir / "confusion_matrix.csv").is_file()
    assert (out_dir / "confusion_matrix.png").is_file()
    assert (out_dir / "errors.csv").is_file()

    saved_metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert saved_metrics["macro_f1"] == summary["macro_f1"]

    with (out_dir / "per_class_metrics.csv").open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == PER_CLASS_COLUMNS
        assert len(list(reader)) == 3

    with (out_dir / "errors.csv").open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ERROR_COLUMNS
        error_rows = list(reader)
        assert len(error_rows) == 1
        assert error_rows[0]["true_class"] == "cardboard"
        assert error_rows[0]["predicted_class"] == "glass"


def test_evaluate_model_missing_weights_raises(tmp_path: Path) -> None:
    """Missing checkpoint fails fast with FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        evaluate_model(
            weights_path=tmp_path / "missing.pt",
            data_config={
                "seed": 42,
                "classes": CLASSES,
                "dataset": {},
                "preprocess": {},
            },
            api_config={"model": {"path": "models/model.onnx", "input_size": 224}},
            root=tmp_path,
        )


def test_export_to_onnx_refuses_overwrite_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existing ONNX file is protected unless overwrite=True is passed."""
    existing = tmp_path / "models" / "model.onnx"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"existing-model")

    class DummyExportModel:
        def export(self, format: str, imgsz: int, simplify: bool = False) -> str:
            exported = tmp_path / "exported.onnx"
            exported.write_bytes(b"new-model")
            return str(exported)

    with pytest.raises(FileExistsError):
        export_to_onnx(DummyExportModel(), existing, input_size=224, num_classes=3)

    monkeypatch.setattr(
        eval_module,
        "verify_onnx_model",
        lambda onnx_path, input_size, num_classes: {
            "onnx_path": str(onnx_path),
            "input_name": "images",
            "input_shape": [1, 3, input_size, input_size],
            "output_shape": [1, num_classes],
            "dummy_inference_ms": 1.2,
        },
    )
    info = export_to_onnx(
        DummyExportModel(), existing, input_size=224, num_classes=3, overwrite=True
    )
    assert info["output_shape"] == [1, 3]
    assert existing.read_bytes() == b"new-model"
