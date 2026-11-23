"""File I/O helpers for configuration and tabular artifacts."""

import csv
import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


def read_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from ``path``."""
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    """Write rows to ``path`` as CSV with a fixed column order."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_csv(
    path: Path,
    row: Mapping[str, Any],
    fieldnames: Sequence[str],
) -> None:
    """Append one row to ``path``, writing the header for a new file."""
    ensure_dir(path.parent)
    needs_header = not path.is_file()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def compute_sha1(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-1 hex digest of a file, read in chunks."""
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_dir(path: Path) -> Path:
    """Create ``path`` including parents and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
