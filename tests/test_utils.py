from pathlib import Path

import pytest

from src.utils.io import compute_sha1, read_yaml, write_csv
from src.utils.logging import setup_logging
from src.utils.seed import set_seed


def test_io_helpers_read_write_and_hash(tmp_path: Path) -> None:
    """Config, CSV, and SHA-1 helpers work on a temporary workspace."""
    config = read_yaml(Path("configs/data.yaml"))
    assert config["seed"] == 42

    target = tmp_path / "nested" / "items.csv"
    write_csv(target, [{"a": 1, "b": "x"}], ["a", "b"])
    assert target.read_text(encoding="utf-8").splitlines() == ["a,b", "1,x"]

    sample = tmp_path / "sample.txt"
    sample.write_bytes(b"abc")
    assert compute_sha1(sample) == "a9993e364706816aba3e25717850c26c9cd0d89d"


def test_missing_config_raises_file_not_found(tmp_path: Path) -> None:
    """Reading a missing YAML file fails explicitly."""
    with pytest.raises(FileNotFoundError):
        read_yaml(tmp_path / "missing.yaml")


def test_invalid_log_level_and_seed_raise_value_error() -> None:
    """Invalid logging and seed inputs are rejected."""
    with pytest.raises(ValueError):
        setup_logging("VERBOSE")
    with pytest.raises(ValueError):
        set_seed(-1)
