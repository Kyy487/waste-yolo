"""End-to-end data preparation pipeline for YOLO-cls."""

import argparse
import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.data.dedup import ImageRecord, RemovedRecord, compute_phash, remove_duplicates
from src.data.split import stratified_split
from src.utils.io import compute_sha1, ensure_dir, read_yaml, write_csv
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


def relative_path(path: Path, root: Path) -> str:
    """Return a POSIX-style path relative to ``root`` when possible."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def scan_raw_images(
    raw_dir: Path,
    classes: Sequence[str],
    valid_extensions: Sequence[str],
    hash_size: int,
) -> tuple[list[ImageRecord], list[RemovedRecord]]:
    """Scan raw class folders, hashing valid images and flagging corrupt ones."""
    extensions = {ext.lower() for ext in valid_extensions}
    records: list[ImageRecord] = []
    removed: list[RemovedRecord] = []

    for class_name in sorted(classes):
        class_dir = raw_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing raw class directory: {class_dir}")
        paths = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        )
        if not paths:
            raise ValueError(f"No images found for class {class_name} in {class_dir}")

        kept = 0
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                removed.append(RemovedRecord(path, class_name, "CORRUPT_IMAGE"))
                continue
            try:
                phash = compute_phash(image, hash_size)
            except (ValueError, cv2.error):
                removed.append(RemovedRecord(path, class_name, "CORRUPT_IMAGE"))
                continue
            records.append(ImageRecord(path, class_name, compute_sha1(path), phash))
            kept += 1
        LOGGER.info("class=%s found=%d usable=%d", class_name, len(paths), kept)

    return records, removed


def preprocess_image(image_bgr: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
    """Resize with optional padding and return an RGB array of fixed size."""
    size = int(config["image_size"])
    if size <= 0:
        raise ValueError("image_size must be positive")

    interpolation_name = str(config.get("resize_interpolation", "INTER_AREA"))
    interpolation = getattr(cv2, interpolation_name, None)
    if interpolation is None:
        raise ValueError(f"Unsupported OpenCV interpolation: {interpolation_name}")

    working = image_bgr
    if bool(config.get("clahe", False)):
        lab = cv2.cvtColor(working, cv2.COLOR_BGR2LAB)
        light, alpha, beta = cv2.split(lab)
        grid = int(config["clahe_tile_grid"])
        light = cv2.createCLAHE(
            clipLimit=float(config["clahe_clip_limit"]),
            tileGridSize=(grid, grid),
        ).apply(light)
        working = cv2.cvtColor(cv2.merge((light, alpha, beta)), cv2.COLOR_LAB2BGR)

    rgb = cv2.cvtColor(working, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    if not bool(config.get("keep_aspect_ratio", True)):
        return cv2.resize(rgb, (size, size), interpolation=interpolation)

    scale = min(size / width, size / height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    resized = cv2.resize(rgb, (new_width, new_height), interpolation=interpolation)
    padding = int(config.get("padding_color", 0))
    canvas = np.full((size, size, 3), padding, dtype=np.uint8)
    top = (size - new_height) // 2
    left = (size - new_width) // 2
    canvas[top : top + new_height, left : left + new_width] = resized
    return canvas


def materialize_dataset(
    splits: Mapping[str, Sequence[ImageRecord]],
    dataset_dir: Path,
    config: Mapping[str, Any],
) -> None:
    """Write preprocessed 224x224 RGB images into YOLO-cls folder layout."""
    quality = int(config["jpeg_quality"])
    for split_name, records in splits.items():
        for record in records:
            image = cv2.imread(str(record.path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Cannot read image: {record.path}")
            rgb = preprocess_image(image, config)
            output_dir = ensure_dir(dataset_dir / split_name / record.class_name)
            output_path = output_dir / f"{record.path.stem}_{record.sha1[:8]}.jpg"
            Image.fromarray(rgb).save(output_path, quality=quality)


def write_audit_report(
    raw_counts: Mapping[str, int],
    final_counts: Mapping[str, int],
    root: Path,
    config: Mapping[str, Any],
) -> None:
    """Save per-class counts and a before/after distribution plot."""
    dataset_config = config["dataset"]
    csv_path = root / dataset_config["audit_csv"]
    plot_path = root / dataset_config["audit_plot"]
    classes = list(config["classes"])
    write_csv(
        csv_path,
        [
            {
                "class": class_name,
                "raw": raw_counts.get(class_name, 0),
                "final": final_counts.get(class_name, 0),
                "removed": raw_counts.get(class_name, 0)
                - final_counts.get(class_name, 0),
            }
            for class_name in classes
        ],
        ["class", "raw", "final", "removed"],
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positions = list(range(len(classes)))
    width = 0.35
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(
        [position - width / 2 for position in positions],
        [raw_counts.get(class_name, 0) for class_name in classes],
        width,
        label="raw",
    )
    axis.bar(
        [position + width / 2 for position in positions],
        [final_counts.get(class_name, 0) for class_name in classes],
        width,
        label="final",
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(classes, rotation=45, ha="right")
    axis.set_ylabel("images")
    axis.set_title("Class distribution before and after deduplication")
    axis.legend()
    figure.tight_layout()
    ensure_dir(plot_path.parent)
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)
    LOGGER.info("audit report written: %s %s", csv_path, plot_path)


def build_dataset(config: Mapping[str, Any], root: Path) -> dict[str, int]:
    """Run scan, dedup, split, and materialization using ``config``."""
    set_seed(int(config["seed"]))
    dataset_config = config["dataset"]
    raw_dir = root / dataset_config["raw_dir"]
    dataset_dir = root / dataset_config["dataset_dir"]

    records, removed = scan_raw_images(
        raw_dir=raw_dir,
        classes=config["classes"],
        valid_extensions=dataset_config["valid_extensions"],
        hash_size=int(config["dedup"]["hash_size"]),
    )
    LOGGER.info("scanned=%d corrupt=%d", len(records), len(removed))
    raw_counts = Counter(record.class_name for record in records)
    raw_counts.update(
        item.class_name for item in removed if item.reason == "CORRUPT_IMAGE"
    )

    dedup_config = config["dedup"]
    if bool(dedup_config.get("enabled", True)):
        records, duplicates = remove_duplicates(
            records,
            threshold=int(dedup_config["hamming_threshold"]),
            hash_blocks=int(dedup_config["hash_blocks"]),
            hash_size=int(dedup_config["hash_size"]),
            drop_conflicting_labels=bool(dedup_config["drop_conflicting_labels"]),
        )
        removed.extend(duplicates)

    write_csv(
        root / dedup_config["removed_csv"],
        [
            {
                "filepath": relative_path(item.path, root),
                "class": item.class_name,
                "reason": item.reason,
                "duplicate_of": (
                    relative_path(item.duplicate_of, root)
                    if item.duplicate_of
                    else None
                ),
            }
            for item in removed
        ],
        ["filepath", "class", "reason", "duplicate_of"],
    )

    splits = stratified_split(records, config["split"], int(config["seed"]))
    write_csv(
        root / dataset_config["splits_csv"],
        [
            {
                "filepath": relative_path(record.path, root),
                "class": record.class_name,
                "split": split_name,
                "sha1": record.sha1,
            }
            for split_name, records_in_split in splits.items()
            for record in records_in_split
        ],
        ["filepath", "class", "split", "sha1"],
    )
    materialize_dataset(splits, dataset_dir, config["preprocess"])
    write_audit_report(
        raw_counts, Counter(record.class_name for record in records), root, config
    )

    counts = {name: len(items) for name, items in splits.items()}
    LOGGER.info("dataset ready: %s", counts)
    return counts


def main() -> None:
    """CLI entry point for the data preparation pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    setup_logging()
    config = read_yaml(args.config)
    build_dataset(config, args.root)


if __name__ == "__main__":
    main()
