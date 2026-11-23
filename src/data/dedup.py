"""Perceptual-hash deduplication for image records."""

import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageRecord:
    """One raw image with its class, content hash, and perceptual hash."""

    path: Path
    class_name: str
    sha1: str
    phash: int


@dataclass(frozen=True)
class RemovedRecord:
    """An image excluded from the final dataset and the reason why."""

    path: Path
    class_name: str
    reason: str
    duplicate_of: Path | None = None


def compute_phash(image: np.ndarray, hash_size: int = 8) -> int:
    """Return the perceptual hash of a BGR image as an integer."""
    if hash_size <= 0:
        raise ValueError("hash_size must be positive")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a BGR image with three channels")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    value = imagehash.phash(Image.fromarray(rgb), hash_size=hash_size)
    return int(str(value), 16)


def find_duplicate_groups(
    records: Sequence[ImageRecord],
    threshold: int = 5,
    hash_blocks: int = 8,
    hash_size: int = 8,
) -> list[list[ImageRecord]]:
    """Group records whose perceptual hashes are within ``threshold`` bits."""
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    if hash_blocks <= threshold:
        raise ValueError("hash_blocks must be greater than threshold")
    if hash_size <= 0:
        raise ValueError("hash_size must be positive")

    ordered = sorted(records, key=lambda record: record.path.as_posix())
    total_bits = hash_size * hash_size
    block_bits = (total_bits + hash_blocks - 1) // hash_blocks
    block_mask = (1 << block_bits) - 1
    parent = {record.path: record.path for record in ordered}

    def find(node: Path) -> Path:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != node:
            parent[node], node = root, parent[node]
        return root

    def union(left: Path, right: Path) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    by_hash: dict[int, list[ImageRecord]] = defaultdict(list)
    for record in ordered:
        by_hash[record.phash].append(record)
    representatives: list[ImageRecord] = []
    for phash in sorted(by_hash):
        exact_group = by_hash[phash]
        for extra in exact_group[1:]:
            union(exact_group[0].path, extra.path)
        representatives.append(exact_group[0])

    buckets: dict[tuple[int, int], list[tuple[int, ImageRecord]]] = defaultdict(list)
    for index, record in enumerate(representatives):
        for block_index in range(hash_blocks):
            shift = block_index * block_bits
            block_value = (record.phash >> shift) & block_mask
            buckets[(block_index, block_value)].append((index, record))

    for bucket in buckets.values():
        for position, (_, left) in enumerate(bucket):
            for _, right in bucket[position + 1 :]:
                if (left.phash ^ right.phash).bit_count() <= threshold:
                    union(left.path, right.path)

    groups: dict[Path, list[ImageRecord]] = defaultdict(list)
    for record in ordered:
        groups[find(record.path)].append(record)
    result = [
        sorted(group, key=lambda item: item.path.as_posix())
        for group in groups.values()
    ]
    return sorted(result, key=lambda group: group[0].path.as_posix())


def remove_duplicates(
    records: Sequence[ImageRecord],
    threshold: int = 5,
    hash_blocks: int = 8,
    hash_size: int = 8,
    drop_conflicting_labels: bool = True,
) -> tuple[list[ImageRecord], list[RemovedRecord]]:
    """Keep one representative per duplicate group and list excluded images."""
    kept: list[ImageRecord] = []
    removed: list[RemovedRecord] = []
    excluded_paths: set[Path] = set()

    for group in find_duplicate_groups(records, threshold, hash_blocks, hash_size):
        labels = {record.class_name for record in group}
        if len(labels) > 1 and drop_conflicting_labels:
            for record in group:
                removed.append(
                    RemovedRecord(record.path, record.class_name, "CONFLICTING_LABELS")
                )
                excluded_paths.add(record.path)
            LOGGER.warning(
                "dropped conflicting duplicate group: %s", [str(p) for p in labels]
            )
            continue

        representative = min(group, key=lambda record: record.path.as_posix())
        for record in group:
            if record.path == representative.path:
                continue
            removed.append(
                RemovedRecord(
                    record.path, record.class_name, "DUPLICATE", representative.path
                )
            )
            excluded_paths.add(record.path)

    kept = [
        record
        for record in sorted(records, key=lambda item: item.path.as_posix())
        if record.path not in excluded_paths
    ]
    LOGGER.info("dedup kept=%d removed=%d", len(kept), len(removed))
    return kept, sorted(removed, key=lambda item: item.path.as_posix())
