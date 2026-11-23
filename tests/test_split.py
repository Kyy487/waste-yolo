from pathlib import Path

import pytest

from src.data.dedup import ImageRecord
from src.data.split import stratified_split


def make_records() -> list[ImageRecord]:
    return [
        ImageRecord(Path(f"{class_name}/{index}.jpg"), class_name, str(index), index)
        for class_name in ("glass", "metal")
        for index in range(20)
    ]


def test_stratified_split_is_deterministic_and_exhaustive() -> None:
    """Splits are reproducible, disjoint, and preserve every class."""
    records = make_records()
    first = stratified_split(records, {"train": 0.70, "val": 0.15, "test": 0.15}, 42)
    second = stratified_split(records, {"train": 0.70, "val": 0.15, "test": 0.15}, 42)

    assert {name: [item.path for item in first[name]] for name in first} == {
        name: [item.path for item in second[name]] for name in second
    }
    all_items = [item.path for items in first.values() for item in items]
    assert len(all_items) == len(records)
    assert len(set(all_items)) == len(records)
    for name, items in first.items():
        assert {item.class_name for item in items} == {"glass", "metal"}


def test_invalid_ratios_raise_value_error() -> None:
    """Ratios that do not sum to one are rejected."""
    with pytest.raises(ValueError):
        stratified_split(make_records(), {"train": 0.7, "val": 0.2, "test": 0.2}, 42)
