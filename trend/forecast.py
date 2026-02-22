# -*- coding: utf-8 -*-
'''
Forecasting / polynomial extension utilities.

Given a training fit (t_hat_train) and drift m_hat, build a global
Δ^d-polynomial that matches the last d training points and enforces
Δ^d t_t = m_hat for all t.
'''
from __future__ import annotations

from math import comb as _comb
import numpy as np


def forecast_trend(t_hat: np.ndarray, d: int, m_hat: float, h: int) -> np.ndarray:
    '''
    h-step-ahead forecast of trend for arbitrary d >= 0.
    '''
    t_hat = np.asarray(t_hat, dtype=float).ravel()
    if h <= 0:
        return np.array([], dtype=float)
    if d < 0:
        raise ValueError("d must be >= 0")

    if d == 0:
        return np.full(h, float(m_hat), dtype=float)

    N = t_hat.size
    d_eff = min(int(d), N)
    last = t_hat[-d_eff:].copy()
    coeffs = np.array([(-1) ** (d_eff - k) * _comb(d_eff, k) for k in range(d_eff)], dtype=float)

    out = np.empty(h, dtype=float)
    for i in range(h):
        next_val = float(m_hat) - float(coeffs @ last)
        out[i] = next_val
        last[:-1] = last[1:]
        last[-1] = next_val
    return out


def build_polynomial_from_train_tail(
    t_hat_train: np.ndarray,
    d: int,
    m_hat: float,
    N_total: int,
    N_train: int,
) -> np.ndarray:
    '''
    Build the global Δ^d polynomial implied by drift m_hat, anchored at the
    right edge of training trend.
    '''
    t_hat_train = np.asarray(t_hat_train, dtype=float).ravel()
    N_train = int(N_train)
    N_total = int(N_total)
    if N_total <= 0:
        return np.array([], dtype=float)

    if d <= 0:
        return np.full(N_total, float(m_hat), dtype=float)

    d_eff = min(int(d), N_train)
    poly = np.empty(N_total, dtype=float)

    j0 = N_train - 1
    start_tail = j0 - d_eff + 1
    poly[start_tail : j0 + 1] = t_hat_train[-d_eff:]

    back_coeffs = np.array([(-1) ** (d_eff - k) * _comb(d_eff, k) for k in range(1, d_eff + 1)], dtype=float)
    coef0 = (-1) ** d_eff

    for j in range(start_tail - 1, -1, -1):
        sum_future = float(np.dot(back_coeffs, poly[j + 1 : j + 1 + d_eff]))
        poly[j] = (float(m_hat) - sum_future) * coef0

    fwd_coeffs = np.array([(-1) ** (d_eff - k) * _comb(d_eff, k) for k in range(d_eff)], dtype=float)
    for i in range(j0 + 1, N_total):
        sum_prev = float(np.dot(fwd_coeffs, poly[i - d_eff : i]))
        poly[i] = float(m_hat) - sum_prev

    return poly
