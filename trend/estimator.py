# -*- coding: utf-8 -*-
'''
High-level estimator: reusable wrapper around the core spectral solver.

Provides:
- fit from s on training data
- validation loss function J(s)
- scan of all local minima of J(s)
- strict polynomial extension across full sample
'''
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List
import numpy as np

from .solver import GuerreroSpectralSolver
from .forecast import forecast_trend, build_polynomial_from_train_tail
from .selection import find_all_local_minima_s
from .arima import forecast_trend_arima, ArimaTrendForecast


@dataclass(frozen=True)
class FitResult:
    s_unit: float
    lam: float
    t_hat: np.ndarray
    m_hat: float
    sigma2_hat: float
    s_unit_real: float


@dataclass(frozen=True)
class MinimaPoint:
    s_unit: float
    val_mse: float


@dataclass(frozen=True)
class MinimaScan:
    d: int
    minima: List[MinimaPoint]
    s_grid: np.ndarray
    J_grid: np.ndarray

    def best(self) -> MinimaPoint:
        if not self.minima:
            raise ValueError("No minima found.")
        return min(self.minima, key=lambda p: p.val_mse)


class GuerreroTrendEstimator:
    def __init__(self, d: int, n_train: int):
        self.d = int(d)
        self.n_train = int(n_train)
        self.solver = GuerreroSpectralSolver(self.n_train, self.d)

    def fit_train(self, Z_train: np.ndarray, s_unit: float) -> FitResult:
        fit, lam = self.solver.fit_for_s(Z_train, s_unit=float(s_unit))
        return FitResult(
            s_unit=float(s_unit),
            lam=float(lam),
            t_hat=np.asarray(fit.t_hat, dtype=float),
            m_hat=float(fit.m_hat),
            sigma2_hat=float(fit.sigma2_hat),
            s_unit_real=float(fit.s_unit_real),
        )

    def validation_loss_fn(self, Z_train: np.ndarray, Z_val: np.ndarray) -> Callable[[float], float]:
        Z_train = np.asarray(Z_train, dtype=float).ravel()
        Z_val = np.asarray(Z_val, dtype=float).ravel()

        if len(Z_val) == 0:
            def J(_s: float) -> float:
                return np.nan
            return J

        def J(s: float) -> float:
            try:
                fit = self.fit_train(Z_train, s_unit=float(s))
            except Exception:
                return np.inf
            t_fore = forecast_trend(fit.t_hat, d=self.d, m_hat=fit.m_hat, h=len(Z_val))
            return float(np.mean((t_fore - Z_val) ** 2))

        return J

    def scan_local_minima(
        self,
        Z_train: np.ndarray,
        Z_val: np.ndarray,
        s_min: float = 1e-3,
        s_max: float = 0.999,
        n_grid: int = 400,
        refine: bool = True,
        refine_iter: int = 25,
    ) -> MinimaScan:
        J = self.validation_loss_fn(Z_train, Z_val)
        out = find_all_local_minima_s(
            J=J,
            s_min=s_min,
            s_max=s_max,
            n_grid=n_grid,
            refine=refine,
            refine_iter=refine_iter,
        )
        minima = [MinimaPoint(s_unit=float(s), val_mse=float(j)) for s, j in zip(out.s_minima, out.J_minima)]
        return MinimaScan(d=self.d, minima=minima, s_grid=out.s_grid, J_grid=out.J_grid)

    def forecast_trend_arima(
        self,
        t_hat_train: np.ndarray,
        h: int,
        order: tuple[int, int, int] = (1, 0, 0),
        seasonal_order: tuple[int, int, int, int] | None = None,
        trend: str | None = "c",
        enforce_stationarity: bool = True,
        enforce_invertibility: bool = True,
    ) -> ArimaTrendForecast:
        '''
        Convenience wrapper: fit ARIMA on the trend estimate (training) and forecast h steps.
        Requires optional dependency: statsmodels (install extra ".[arima]").
        '''
        return forecast_trend_arima(
            t_hat_train=t_hat_train,
            h=int(h),
            order=order,
            seasonal_order=seasonal_order,
            trend=trend,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
        )

    def build_polynomial_full(self, t_hat_train: np.ndarray, m_hat: float, n_total: int, n_train: int) -> np.ndarray:
        return build_polynomial_from_train_tail(
            t_hat_train=t_hat_train,
            d=self.d,
            m_hat=float(m_hat),
            N_total=int(n_total),
            N_train=int(n_train),
        )
