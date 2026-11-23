"""YOLO-cls training entry point with reproducible run artifacts."""

import argparse
import csv
import json
import logging
import platform
import subprocess
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.utils import LOGGER as ULTRALYTICS_LOGGER

from src.utils.io import append_csv, ensure_dir, read_yaml
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
EXPERIMENT_COLUMNS = [
    "run_id",
    "model",
    "imgsz",
    "epochs",
    "batch",
    "aug",
    "imbalance",
    "val_acc",
    "val_macro_f1",
    "train_time_min",
    "notes",
]


def commit_hash(root: Path) -> str:
    """Return the current git commit, or ``unknown`` outside a repository."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def library_versions() -> dict[str, str]:
    """Return versions of the libraries that affect reproducibility."""
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
    }


def macro_f1_from_confusion_matrix(matrix: np.ndarray) -> float:
    """Compute macro-F1 from a classifier confusion matrix.

    Ultralytics stores predicted classes in rows and true classes in columns.
    """
    array = np.asarray(matrix)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("confusion matrix must be square")
    if array.sum() == 0:
        return 0.0

    true_positive = np.diag(array).astype(float)
    predicted = array.sum(axis=1).astype(float)
    actual = array.sum(axis=0).astype(float)
    precision = np.divide(
        true_positive, predicted, out=np.zeros_like(true_positive), where=predicted > 0
    )
    recall = np.divide(
        true_positive, actual, out=np.zeros_like(true_positive), where=actual > 0
    )
    denominator = precision + recall
    f1 = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0,
    )
    return float(f1.mean())


def validation_metrics(metrics: Any | None) -> dict[str, float | None]:
    """Extract validation top-1 accuracy and macro-F1 from Ultralytics metrics."""
    if metrics is None:
        return {"val_acc": None, "val_macro_f1": None}
    accuracy = float(metrics.results_dict["metrics/accuracy_top1"])
    matrix = np.asarray(metrics.confusion_matrix.matrix)
    return {
        "val_acc": accuracy,
        "val_macro_f1": macro_f1_from_confusion_matrix(matrix),
    }


def assert_new_run_id(csv_path: Path, run_id: str) -> None:
    """Reject a run ID already recorded in the experiment table."""
    if not csv_path.is_file():
        return
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        existing = {row.get("run_id") for row in csv.DictReader(handle)}
    if run_id in existing:
        raise ValueError(f"run_id already exists in {csv_path}: {run_id}")


def resolve_weights(weights: str, root: Path) -> str:
    """Use a local checkpoint when present, otherwise let Ultralytics fetch it."""
    candidate = root / weights
    return str(candidate) if candidate.is_file() else weights


def train_model(
    config: Mapping[str, Any],
    root: Path,
    config_path: Path,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train one YOLO-cls run and persist its configuration and metadata."""
    options = dict(overrides or {})
    data_config = read_yaml(root / config["dataset"]["config"])
    seed = int(config["training"]["seed"])
    if seed != int(data_config["seed"]):
        raise ValueError("training seed must match data config seed")
    set_seed(seed)

    run_id = str(options.get("run_id", config["run"]["id"]))
    output = config["output"]
    experiments_csv = root / output["experiments_csv"]
    if not options.get("no_record", False):
        assert_new_run_id(experiments_csv, run_id)

    run_dir = root / output["run_dir"] / run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    ensure_dir(run_dir)
    (run_dir / "config.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    LOGGER.addHandler(handler)
    ULTRALYTICS_LOGGER.addHandler(handler)

    training = config["training"]
    epochs = int(options.get("epochs", training["epochs"]))
    imgsz = int(options.get("imgsz", training["imgsz"]))
    batch = int(options.get("batch", training["batch"]))
    patience = int(options.get("patience", training["patience"]))
    fraction = float(options.get("fraction", training["fraction"]))
    device_option = options.get("device", training["device"])
    device = None if str(device_option).lower() == "auto" else device_option
    run_validation = bool(options.get("val", training["val"]))

    weights = str(options.get("weights", config["model"]["weights"]))
    architecture = str(options.get("architecture", config["model"]["architecture"]))
    augmentation = str(options.get("aug", training["augmentation"]))
    imbalance = str(options.get("imbalance", training["imbalance"]))
    notes = str(options.get("notes", config["run"]["notes"]))

    try:
        LOGGER.info(
            "run=%s weights=%s epochs=%d imgsz=%d batch=%d",
            run_id,
            weights,
            epochs,
            imgsz,
            batch,
        )
        model = YOLO(resolve_weights(weights, root))
        started = time.perf_counter()
        model.train(
            data=str(root / config["dataset"]["dataset_dir"]),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=patience,
            seed=seed,
            device=device,
            fraction=fraction,
            project=str(root / output["project"]),
            name=run_id,
            val=run_validation,
            plots=bool(output["plots"]),
            exist_ok=False,
        )
        duration_min = (time.perf_counter() - started) / 60.0
        metric_values = validation_metrics(getattr(model, "metrics", None))
        checkpoint = str(getattr(model, "ckpt_path", ""))
    finally:
        LOGGER.removeHandler(handler)
        ULTRALYTICS_LOGGER.removeHandler(handler)
        handler.close()

    metadata = {
        "run_id": run_id,
        "architecture": architecture,
        "weights": weights,
        "seed": seed,
        "commit": commit_hash(root),
        "versions": library_versions(),
        "duration_min": round(duration_min, 3),
        "metrics": metric_values,
        "checkpoint": checkpoint,
        "options": {key: value for key, value in options.items() if key != "no_record"},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if not options.get("no_record", False):
        append_csv(
            experiments_csv,
            {
                "run_id": run_id,
                "model": architecture,
                "imgsz": imgsz,
                "epochs": epochs,
                "batch": batch,
                "aug": augmentation,
                "imbalance": imbalance,
                "val_acc": (
                    round(metric_values["val_acc"], 6)
                    if metric_values["val_acc"] is not None
                    else ""
                ),
                "val_macro_f1": (
                    round(metric_values["val_macro_f1"], 6)
                    if metric_values["val_macro_f1"] is not None
                    else ""
                ),
                "train_time_min": round(duration_min, 2),
                "notes": notes,
            },
            EXPERIMENT_COLUMNS,
        )

    LOGGER.info(
        "run=%s finished in %.2f min metrics=%s", run_id, duration_min, metric_values
    )
    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "duration_min": duration_min,
        **metric_values,
    }


def main() -> None:
    """CLI entry point for a single training run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id")
    parser.add_argument("--weights")
    parser.add_argument("--architecture")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--fraction", type=float)
    parser.add_argument("--device")
    parser.add_argument("--aug")
    parser.add_argument("--imbalance")
    parser.add_argument("--notes")
    parser.add_argument("--no-val", action="store_true")
    parser.add_argument("--no-record", action="store_true")
    args = parser.parse_args()

    setup_logging()
    root = args.root.resolve()
    config_path = (
        args.config if args.config.is_absolute() else (root / args.config).resolve()
    )
    options: dict[str, Any] = {}
    for name in (
        "run_id",
        "weights",
        "architecture",
        "epochs",
        "imgsz",
        "batch",
        "patience",
        "fraction",
        "device",
        "aug",
        "imbalance",
        "notes",
    ):
        value = getattr(args, name)
        if value is not None:
            options[name] = value
    options["val"] = not args.no_val
    options["no_record"] = args.no_record

    train_model(read_yaml(config_path), root, config_path, options)


if __name__ == "__main__":
    main()
