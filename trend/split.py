# -*- coding: utf-8 -*-
'''Train/validation/test split utilities for 1D series.'''
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SplitInfo:
    n_train: int
    n_val: int
    n_test: int


def train_val_test_split(
    Z: np.ndarray,
    frac_train: float = 0.6,
    frac_val: float = 0.2,
    min_train: int = 50,
    min_val: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, SplitInfo]:
    '''
    3-way split of 1D series Z into train/val/test.
    '''
    Z = np.asarray(Z, dtype=float).ravel()
    N = int(Z.size)
    if N < 3:
        raise ValueError("Need at least 3 observations.")

    n_train = max(int(min_train), int(frac_train * N))
    n_val = max(int(min_val), int(frac_val * N))

    if n_train + n_val >= N:
        n_val = max(1, N - 1 - n_train)
        if n_val < min_val:
            n_train = max(min_train, N - 1 - n_val)

    n_test = N - n_train - n_val
    if n_test < 1:
        n_test = 1
        n_val = 0
        n_train = N - n_test

    Z_train = Z[:n_train]
    Z_val = Z[n_train : n_train + n_val]
    Z_test = Z[n_train + n_val :]

    return Z_train, Z_val, Z_test, SplitInfo(n_train=n_train, n_val=n_val, n_test=n_test)
