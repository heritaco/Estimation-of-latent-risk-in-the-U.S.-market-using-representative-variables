# -*- coding: utf-8 -*-
'''Matplotlib plot helpers (kept separate from core solver).'''
from __future__ import annotations

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt


def plot_scan(local_minima, title: str = "", ax: Optional["plt.Axes"] = None):
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(local_minima.s_grid, local_minima.J_grid, linewidth=1.5, label="J(s) (val MSE)")
    ax.scatter(local_minima.s_minima, local_minima.J_minima, marker="*", s=70, color="k", label="local minima")
    ax.set_xlabel("s")
    ax.set_ylabel("J(s)")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend()
    return ax


def plot_train_val_test_fit(
    Z_all: np.ndarray,
    n_train: int,
    n_val: int,
    t_hat_train: np.ndarray,
    poly_full: np.ndarray,
    label: str = "series",
    title: str = "",
    ax: Optional["plt.Axes"] = None,
):
    Z_all = np.asarray(Z_all, dtype=float).ravel()
    poly_full = np.asarray(poly_full, dtype=float).ravel()
    t_hat_train = np.asarray(t_hat_train, dtype=float).ravel()

    N = len(Z_all)
    idx_all = np.arange(N)
    idx_train = idx_all[:n_train]
    idx_val = idx_all[n_train : n_train + n_val]
    idx_test = idx_all[n_train + n_val :]

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(idx_train, Z_all[idx_train], linewidth=1.0, label=f"{label} (train)")
    if len(idx_val) > 0:
        ax.plot(idx_val, Z_all[idx_val], linewidth=1.0, label=f"{label} (val)")
    if len(idx_test) > 0:
        ax.plot(idx_test, Z_all[idx_test], linewidth=1.0, label=f"{label} (test)")

    ax.plot(idx_train, t_hat_train, linewidth=2.0, label="trend fit (train)")
    ax.plot(idx_all, poly_full, linestyle="--", linewidth=2.0, label="Δ^d polynomial")

    ax.axvline(n_train - 0.5, color="k", linestyle="--", linewidth=1)
    if n_val > 0:
        ax.axvline(n_train + n_val - 0.5, color="k", linestyle=":", linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel(label)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(fontsize=8)
    return ax
