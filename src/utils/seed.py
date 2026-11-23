"""Deterministic random seed helpers."""

import logging
import random

import numpy as np

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random generators."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        LOGGER.warning("PyTorch is not installed; skipping torch seed.")
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
