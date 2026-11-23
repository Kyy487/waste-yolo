"""Stratified dataset splitting with a fixed seed."""

import logging
from collections.abc import Mapping, Sequence

from sklearn.model_selection import train_test_split

from src.data.dedup import ImageRecord

LOGGER = logging.getLogger(__name__)
SPLIT_NAMES = ("train", "val", "test")


def stratified_split(
    records: Sequence[ImageRecord],
    ratios: Mapping[str, float],
    seed: int = 42,
) -> dict[str, list[ImageRecord]]:
    """Split records into train/val/test while preserving class proportions."""
    if set(ratios) != set(SPLIT_NAMES):
        raise ValueError(f"ratios must contain exactly {set(SPLIT_NAMES)}")
    values = [float(ratios[name]) for name in SPLIT_NAMES]
    if any(value <= 0 for value in values) or abs(sum(values) - 1.0) > 1e-6:
        raise ValueError("split ratios must be positive and sum to 1.0")

    ordered = sorted(records, key=lambda record: record.path.as_posix())
    counts: dict[str, int] = {}
    for record in ordered:
        counts[record.class_name] = counts.get(record.class_name, 0) + 1
    if any(count < 2 for count in counts.values()):
        raise ValueError("each class needs at least two images for a stratified split")

    labels = [record.class_name for record in ordered]
    train_val, test = train_test_split(
        ordered,
        test_size=values[2],
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    remaining = values[0] + values[1]
    val_fraction = values[1] / remaining
    train, val = train_test_split(
        train_val,
        test_size=val_fraction,
        random_state=seed,
        shuffle=True,
        stratify=[record.class_name for record in train_val],
    )

    splits = {"train": train, "val": val, "test": test}
    for name, items in splits.items():
        LOGGER.info("split=%s count=%d", name, len(items))
    return {
        name: sorted(items, key=lambda record: record.path.as_posix())
        for name, items in splits.items()
    }
