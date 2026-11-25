"""Model evaluation, error analysis, and ONNX export for YOLO-cls."""

import argparse
import copy
import json
import logging
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import onnxruntime as ort
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from ultralytics import YOLO

from src.utils.io import ensure_dir, read_yaml, write_csv
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LOGGER = logging.getLogger(__name__)

PER_CLASS_COLUMNS = ["class", "precision", "recall", "f1", "support"]
ERROR_COLUMNS = [
    "filepath",
    "true_class",
    "predicted_class",
    "confidence",
    "true_confidence",
]


@dataclass(frozen=True)
class EvalSample:
    """One labeled image scheduled for evaluation."""

    path: Path
    true_label: str
    true_index: int


@dataclass(frozen=True)
class PredictionRecord:
    """Predicted class and confidence scores for one evaluated image."""

    filepath: str
    true_label: str
    true_index: int
    predicted_label: str
    predicted_index: int
    confidence: float
    true_confidence: float


def relative_path(path: Path, root: Path) -> str:
    """Return a POSIX-style path relative to ``root`` when possible."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_split_dir(data_config: Mapping[str, Any], root: Path, split: str) -> Path:
    """Resolve the folder path for ``val``, ``test``, or ``realworld`` split."""
    dataset_cfg = data_config["dataset"]
    if split in ("val", "test"):
        return root / dataset_cfg["dataset_dir"] / split
    if split == "realworld":
        return root / dataset_cfg["realworld_dir"]
    raise ValueError(f"Unsupported split: {split}. Use 'val', 'test', or 'realworld'.")


def collect_eval_samples(
    split_dir: Path,
    classes: Sequence[str],
    valid_extensions: Sequence[str],
    limit: int | None = None,
) -> list[EvalSample]:
    """Collect labeled image paths in fixed class order, optionally capped."""
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer")

    extensions = {ext.lower() for ext in valid_extensions}
    per_class_samples: list[list[EvalSample]] = []

    for index, class_name in enumerate(classes):
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            LOGGER.warning("Missing class folder in %s: %s", split_dir, class_name)
            per_class_samples.append([])
            continue
        paths = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        )
        per_class_samples.append(
            [
                EvalSample(path=path, true_label=class_name, true_index=index)
                for path in paths
            ]
        )

    if limit is None:
        samples = [sample for bucket in per_class_samples for sample in bucket]
    else:
        # Round-robin across classes so small smoke evaluations cover every class.
        samples = []
        cursor = 0
        while len(samples) < limit:
            added = False
            for bucket in per_class_samples:
                if cursor < len(bucket):
                    samples.append(bucket[cursor])
                    added = True
                    if len(samples) >= limit:
                        break
            if not added:
                break
            cursor += 1

    if not samples:
        raise ValueError(f"No valid images found in {split_dir}")
    return samples


def validate_model_classes(model_names: Any, expected_classes: Sequence[str]) -> None:
    """Ensure the model's class index order matches the project contract."""
    if model_names is None:
        return
    if isinstance(model_names, Mapping):
        ordered = [str(model_names[idx]) for idx in sorted(model_names)]
    else:
        ordered = [str(name) for name in model_names]
    if ordered != list(expected_classes):
        raise ValueError(
            f"Model class order mismatch: got {ordered}, expected {list(expected_classes)}"
        )


def predict_samples(
    model: Any,
    samples: Sequence[EvalSample],
    classes: Sequence[str],
    root: Path,
    imgsz: int,
    batch_size: int = 32,
    device: str | None = None,
) -> tuple[list[PredictionRecord], float]:
    """Run batched YOLO-cls inference and return predictions with avg ms/image."""
    if not samples:
        raise ValueError("samples must not be empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    validate_model_classes(getattr(model, "names", None), classes)

    records: list[PredictionRecord] = []
    started = time.perf_counter()

    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        sources = [str(sample.path) for sample in batch]
        results = model.predict(
            source=sources,
            imgsz=imgsz,
            batch=len(batch),
            device=device,
            verbose=False,
        )
        if len(results) != len(batch):
            raise RuntimeError(
                f"Expected {len(batch)} prediction results, got {len(results)}"
            )
        for sample, result in zip(batch, results, strict=True):
            probs_raw = result.probs.data
            if hasattr(probs_raw, "cpu"):
                probs = np.asarray(probs_raw.cpu().numpy(), dtype=float)
            else:
                probs = np.asarray(probs_raw, dtype=float)
            if probs.shape != (len(classes),):
                raise ValueError(
                    f"Expected probability vector of length {len(classes)}, got {probs.shape}"
                )
            pred_index = int(np.argmax(probs))
            records.append(
                PredictionRecord(
                    filepath=relative_path(sample.path, root),
                    true_label=sample.true_label,
                    true_index=sample.true_index,
                    predicted_label=str(classes[pred_index]),
                    predicted_index=pred_index,
                    confidence=float(probs[pred_index]),
                    true_confidence=float(probs[sample.true_index]),
                )
            )

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    avg_ms = elapsed_ms / len(samples)
    return records, avg_ms


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    classes: Sequence[str],
) -> dict[str, Any]:
    """Compute overall accuracy, macro-F1, per-class metrics, and confusion matrix."""
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be non-empty and have equal length")

    labels = list(range(len(classes)))
    accuracy = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    per_class: list[dict[str, Any]] = []
    for idx, class_name in enumerate(classes):
        per_class.append(
            {
                "class": str(class_name),
                "precision": round(float(precision[idx]), 6),
                "recall": round(float(recall[idx]), 6),
                "f1": round(float(f1[idx]), 6),
                "support": int(support[idx]),
            }
        )

    worst_idx = int(np.argmin(f1))
    num_errors = int(sum(int(t != p) for t, p in zip(y_true, y_pred, strict=True)))

    return {
        "num_samples": len(y_true),
        "num_errors": num_errors,
        "accuracy": round(accuracy, 6),
        "macro_precision": round(float(np.mean(precision)), 6),
        "macro_recall": round(float(np.mean(recall)), 6),
        "macro_f1": round(float(np.mean(f1)), 6),
        "min_class_f1": round(float(f1[worst_idx]), 6),
        "worst_class": str(classes[worst_idx]),
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def extract_error_rows(records: Sequence[PredictionRecord]) -> list[dict[str, Any]]:
    """Return misclassified samples sorted by descending confidence."""
    errors = [
        {
            "filepath": rec.filepath,
            "true_class": rec.true_label,
            "predicted_class": rec.predicted_label,
            "confidence": round(rec.confidence, 6),
            "true_confidence": round(rec.true_confidence, 6),
        }
        for rec in records
        if rec.predicted_index != rec.true_index
    ]
    errors.sort(key=lambda row: (-float(row["confidence"]), str(row["filepath"])))
    return errors


def save_confusion_matrix_csv(
    matrix: np.ndarray, classes: Sequence[str], path: Path
) -> None:
    """Save the confusion matrix (rows=true, cols=predicted) as CSV."""
    fieldnames = ["true_class", *classes]
    rows: list[dict[str, Any]] = []
    for row_idx, class_name in enumerate(classes):
        row: dict[str, Any] = {"true_class": class_name}
        for col_idx, pred_name in enumerate(classes):
            row[pred_name] = int(matrix[row_idx, col_idx])
        rows.append(row)
    write_csv(path, rows, fieldnames)


def plot_confusion_matrix(
    matrix: np.ndarray,
    classes: Sequence[str],
    output_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """Render and save a labeled confusion matrix heatmap as PNG."""
    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ticks = np.arange(len(classes))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)

    threshold = matrix.max() / 2.0 if matrix.size > 0 else 0.0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = int(matrix[row, col])
            color = "white" if value > threshold else "black"
            ax.text(
                col, row, str(value), ha="center", va="center", color=color, fontsize=9
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def verify_onnx_model(
    onnx_path: Path, input_size: int, num_classes: int
) -> dict[str, Any]:
    """Verify ONNX input/output contract with a dummy CPU forward pass."""
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    dummy = np.zeros((1, 3, input_size, input_size), dtype=np.float32)

    started = time.perf_counter()
    outputs = session.run(None, {input_meta.name: dummy})
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    output_array = np.asarray(outputs[0])
    if output_array.shape != (1, num_classes):
        raise ValueError(
            f"ONNX output shape mismatch: got {output_array.shape}, expected (1, {num_classes})"
        )

    return {
        "onnx_path": str(onnx_path),
        "input_name": input_meta.name,
        "input_shape": [1, 3, input_size, input_size],
        "output_shape": list(output_array.shape),
        "dummy_inference_ms": round(elapsed_ms, 3),
    }


def export_to_onnx(
    model: Any,
    onnx_path: Path,
    input_size: int,
    num_classes: int,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Export a YOLO-cls checkpoint to ONNX and verify its I/O contract."""
    if onnx_path.exists() and not overwrite:
        raise FileExistsError(
            f"ONNX file already exists: {onnx_path}. Pass --overwrite-onnx or choose a new path."
        )
    ensure_dir(onnx_path.parent)
    try:
        exported = Path(model.export(format="onnx", imgsz=input_size, simplify=False))
    except ModuleNotFoundError as exc:
        if "onnxscript" not in str(exc):
            raise
        LOGGER.warning(
            "Ultralytics ONNX export hit missing onnxscript on newer PyTorch; "
            "falling back to torch.onnx.export(dynamo=False)."
        )
        pytorch_model = copy.deepcopy(model.model).eval().cpu()
        dummy = torch.zeros((1, 3, input_size, input_size), dtype=torch.float32)
        torch.onnx.export(
            pytorch_model,
            dummy,
            str(onnx_path),
            verbose=False,
            opset_version=12,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["output0"],
            dynamo=False,
        )
        exported = onnx_path

    if not exported.is_file():
        raise FileNotFoundError(
            f"Ultralytics export did not produce a file: {exported}"
        )

    if exported.resolve() != onnx_path.resolve():
        shutil.copy2(exported, onnx_path)

    info = verify_onnx_model(onnx_path, input_size=input_size, num_classes=num_classes)
    LOGGER.info(
        "Exported and verified ONNX model at %s (output=%s, dummy=%.2f ms)",
        onnx_path,
        info["output_shape"],
        info["dummy_inference_ms"],
    )
    return info


def evaluate_model(
    weights_path: Path,
    data_config: Mapping[str, Any],
    api_config: Mapping[str, Any],
    root: Path,
    split: str = "val",
    output_dir: Path | None = None,
    imgsz: int | None = None,
    batch_size: int = 32,
    device: str | None = None,
    limit: int | None = None,
    do_export_onnx: bool = False,
    onnx_path: Path | None = None,
    overwrite_onnx: bool = False,
    model: Any | None = None,
) -> dict[str, Any]:
    """Evaluate a trained YOLO-cls model and persist metrics, plots, and errors."""
    if not weights_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    seed = int(data_config["seed"])
    set_seed(seed)

    classes = [str(name) for name in data_config["classes"]]
    valid_extensions = [str(ext) for ext in data_config["dataset"]["valid_extensions"]]
    eval_imgsz = int(
        imgsz if imgsz is not None else data_config["preprocess"]["image_size"]
    )
    split_dir = resolve_split_dir(data_config, root, split)

    samples = collect_eval_samples(
        split_dir=split_dir,
        classes=classes,
        valid_extensions=valid_extensions,
        limit=limit,
    )
    LOGGER.info(
        "Evaluating split=%s samples=%d imgsz=%d weights=%s",
        split,
        len(samples),
        eval_imgsz,
        weights_path,
    )

    loaded_model = model if model is not None else YOLO(str(weights_path))
    records, avg_inference_ms = predict_samples(
        model=loaded_model,
        samples=samples,
        classes=classes,
        root=root,
        imgsz=eval_imgsz,
        batch_size=batch_size,
        device=device,
    )

    y_true = [rec.true_index for rec in records]
    y_pred = [rec.predicted_index for rec in records]
    metrics = compute_classification_metrics(y_true, y_pred, classes)
    error_rows = extract_error_rows(records)

    target_dir = (
        output_dir if output_dir is not None else root / "results" / "eval" / split
    )
    ensure_dir(target_dir)

    write_csv(
        target_dir / "per_class_metrics.csv", metrics["per_class"], PER_CLASS_COLUMNS
    )
    write_csv(target_dir / "errors.csv", error_rows, ERROR_COLUMNS)
    save_confusion_matrix_csv(
        metrics["confusion_matrix"], classes, target_dir / "confusion_matrix.csv"
    )
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        classes,
        target_dir / "confusion_matrix.png",
        title=f"Confusion Matrix ({split})",
    )

    summary: dict[str, Any] = {
        "split": split,
        "weights": relative_path(weights_path, root),
        "imgsz": eval_imgsz,
        "seed": seed,
        "num_samples": metrics["num_samples"],
        "num_errors": metrics["num_errors"],
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "min_class_f1": metrics["min_class_f1"],
        "worst_class": metrics["worst_class"],
        "avg_inference_ms": round(avg_inference_ms, 3),
        "per_class": metrics["per_class"],
    }

    if do_export_onnx:
        target_onnx = (
            onnx_path
            if onnx_path is not None
            else root / str(api_config["model"]["path"])
        )
        onnx_size = int(api_config["model"]["input_size"])
        summary["onnx_export"] = export_to_onnx(
            model=loaded_model,
            onnx_path=target_onnx,
            input_size=onnx_size,
            num_classes=len(classes),
            overwrite=overwrite_onnx,
        )

    (target_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    LOGGER.info(
        "Evaluation done: split=%s acc=%.4f macro_f1=%.4f min_f1=%.4f (%s) errors=%d/%d",
        split,
        summary["accuracy"],
        summary["macro_f1"],
        summary["min_class_f1"],
        summary["worst_class"],
        summary["num_errors"],
        summary["num_samples"],
    )
    return summary


def main() -> None:
    """CLI entry point for model evaluation and optional ONNX export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights", type=Path, required=True, help="Path to trained .pt weights"
    )
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--api-config", type=Path, default=Path("configs/api.yaml"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--split",
        choices=("val", "test", "realworld"),
        default="val",
        help="Dataset split to evaluate (default: val; keep test locked until final model)",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device")
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional cap on total images evaluated (round-robin across classes)",
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export the evaluated model to ONNX and verify I/O shape",
    )
    parser.add_argument("--onnx-path", type=Path)
    parser.add_argument("--overwrite-onnx", action="store_true")
    args = parser.parse_args()

    setup_logging()
    root = args.root.resolve()
    data_config_path = (
        args.data_config
        if args.data_config.is_absolute()
        else (root / args.data_config).resolve()
    )
    api_config_path = (
        args.api_config
        if args.api_config.is_absolute()
        else (root / args.api_config).resolve()
    )
    weights_path = (
        args.weights if args.weights.is_absolute() else (root / args.weights).resolve()
    )
    output_dir = (
        None
        if args.output_dir is None
        else (
            args.output_dir
            if args.output_dir.is_absolute()
            else (root / args.output_dir).resolve()
        )
    )
    onnx_path = (
        None
        if args.onnx_path is None
        else (
            args.onnx_path
            if args.onnx_path.is_absolute()
            else (root / args.onnx_path).resolve()
        )
    )
    device = (
        None
        if args.device is None or str(args.device).lower() == "auto"
        else args.device
    )

    evaluate_model(
        weights_path=weights_path,
        data_config=read_yaml(data_config_path),
        api_config=read_yaml(api_config_path),
        root=root,
        split=args.split,
        output_dir=output_dir,
        imgsz=args.imgsz,
        batch_size=args.batch,
        device=device,
        limit=args.limit,
        do_export_onnx=args.export_onnx,
        onnx_path=onnx_path,
        overwrite_onnx=args.overwrite_onnx,
    )


if __name__ == "__main__":
    main()
