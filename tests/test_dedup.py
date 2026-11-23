from pathlib import Path

import numpy as np
import pytest

from src.data.dedup import (
    ImageRecord,
    compute_phash,
    find_duplicate_groups,
    remove_duplicates,
)


def record(name: str, phash: int, class_name: str = "glass") -> ImageRecord:
    return ImageRecord(Path(name), class_name, name, phash)


def test_find_duplicate_groups_uses_hamming_distance() -> None:
    """Near-identical hashes group together while distant ones stay apart."""
    records = [record("a.jpg", 0), record("b.jpg", 3), record("c.jpg", 0xFF00)]
    groups = find_duplicate_groups(records, threshold=2)
    assert [[item.path.name for item in group] for group in groups] == [
        ["a.jpg", "b.jpg"],
        ["c.jpg"],
    ]


def test_remove_duplicates_keeps_lexicographically_first() -> None:
    """Duplicate images are removed in favor of a deterministic representative."""
    kept, removed = remove_duplicates([record("b.jpg", 42), record("a.jpg", 42)])
    assert [item.path.name for item in kept] == ["a.jpg"]
    assert [(item.path.name, item.reason) for item in removed] == [
        ("b.jpg", "DUPLICATE")
    ]


def test_conflicting_labels_are_dropped() -> None:
    """A duplicate pair with different labels is excluded entirely."""
    records = [record("a.jpg", 7, "glass"), record("b.jpg", 7, "metal")]
    kept, removed = remove_duplicates(records, drop_conflicting_labels=True)
    assert kept == []
    assert {item.reason for item in removed} == {"CONFLICTING_LABELS"}


def test_invalid_dedup_parameters_raise_value_error() -> None:
    """Block count must exceed the Hamming threshold."""
    with pytest.raises(ValueError):
        find_duplicate_groups([record("a.jpg", 0)], threshold=8, hash_blocks=8)


def test_compute_phash_is_stable_and_rejects_bad_shape() -> None:
    """Identical images hash identically and invalid arrays are rejected."""
    image = np.full((32, 32, 3), 120, dtype=np.uint8)
    assert compute_phash(image) == compute_phash(image)
    with pytest.raises(ValueError):
        compute_phash(np.zeros((32, 32), dtype=np.uint8))
