# -*- coding: utf-8 -*-
'''
Scan validation loss J(s) over s∈(0,1), detect all local minima, and refine.
'''
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple
import numpy as np


def golden_local(J: Callable[[float], float], a: float, b: float, n_iter: int = 20) -> Tuple[float, float]:
    '''
    Golden-section search restricted to [a,b], assuming J is unimodal.
    Returns (s_star, J(s_star)).
    '''
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    invphi = 1.0 / phi

    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    f_c = J(c)
    f_d = J(d)

    for _ in range(n_iter):
        if f_c < f_d:
            b, d, f_d = d, c, f_c
            c = b - invphi * (b - a)
            f_c = J(c)
        else:
            a, c, f_c = c, d, f_d
            d = a + invphi * (b - a)
            f_d = J(d)

    s_star = 0.5 * (a + b)
    return s_star, float(J(s_star))


@dataclass(frozen=True)
class LocalMinima:
    s_minima: np.ndarray
    J_minima: np.ndarray
    s_grid: np.ndarray
    J_grid: np.ndarray


def find_all_local_minima_s(
    J: Callable[[float], float],
    s_min: float = 1e-3,
    s_max: float = 0.999,
    n_grid: int = 300,
    refine: bool = True,
    refine_iter: int = 20,
) -> LocalMinima:
    '''
    Approximate all local minima of J(s) on [s_min, s_max].
    '''
    s_grid = np.linspace(float(s_min), float(s_max), int(n_grid))
    J_grid = np.array([J(float(s)) for s in s_grid], dtype=float)

    J_cmp = J_grid.copy()
    J_cmp[~np.isfinite(J_cmp)] = np.inf

    idx_candidates = []
    for i in range(1, len(s_grid) - 1):
        if np.isfinite(J_cmp[i]) and (J_cmp[i] <= J_cmp[i - 1]) and (J_cmp[i] <= J_cmp[i + 1]):
            idx_candidates.append(i)
    if np.isfinite(J_cmp[0]) and J_cmp[0] <= J_cmp[1]:
        idx_candidates.append(0)
    if np.isfinite(J_cmp[-1]) and J_cmp[-1] <= J_cmp[-2]:
        idx_candidates.append(len(s_grid) - 1)

    s_list, J_list = [], []
    for idx in idx_candidates:
        if refine:
            a = s_grid[max(0, idx - 1)]
            b = s_grid[min(len(s_grid) - 1, idx + 1)]
            s_star, J_star = golden_local(J, float(a), float(b), n_iter=int(refine_iter))
        else:
            s_star, J_star = float(s_grid[idx]), float(J_grid[idx])

        s_list.append(s_star)
        J_list.append(J_star)

    s_minima = np.array(s_list, dtype=float)
    J_minima = np.array(J_list, dtype=float)
    order = np.argsort(s_minima)
    s_minima, J_minima = s_minima[order], J_minima[order]

    return LocalMinima(
        s_minima=s_minima,
        J_minima=J_minima,
        s_grid=s_grid,
        J_grid=J_grid,
    )
