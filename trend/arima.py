# -*- coding: utf-8 -*-
'''
ARIMA forecasting utilities *on the estimated trend*.

This module is optional: it requires `statsmodels`.
Install with:
  pip install -e ".[arima]"

Core use-case:
  - Fit Guerrero trend on training data -> t_hat_train
  - Fit ARIMA(p,d,q) to t_hat_train
  - Forecast h steps ahead for validation/test

Notation:
  - y_t = t_hat_train[t] (trend estimate)
  - ARIMA(p,d,q): Δ^d y_t = c + φ(L)^{-1} θ(L) ε_t, with ε_t ~ white noise
'''
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class ArimaTrendForecast:
    forecast: np.ndarray
    order: Tuple[int, int, int]
    aic: Optional[float] = None
    bic: Optional[float] = None


def _require_statsmodels():
    try:
        from statsmodels.tsa.arima.model import ARIMA  # noqa: F401
    except Exception as e:
        raise ImportError(
            "statsmodels is required for ARIMA forecasting. "
            "Install with: pip install guerrero-trend[arima]"
        ) from e


def forecast_trend_arima(
    t_hat_train: np.ndarray,
    h: int,
    order: Tuple[int, int, int] = (1, 0, 0),
    seasonal_order: Optional[Tuple[int, int, int, int]] = None,
    trend: Optional[str] = "c",
    enforce_stationarity: bool = True,
    enforce_invertibility: bool = True,
) -> ArimaTrendForecast:
    '''
    Fit ARIMA to the trend estimate and forecast h steps.

    Parameters
    ----------
    t_hat_train : np.ndarray
        Trend estimate on training sample (1D).
    h : int
        Forecast horizon.
    order : (p,d,q)
        ARIMA order.
    seasonal_order : (P,D,Q,s) or None
        If provided, uses SARIMAX backend via ARIMA(..., seasonal_order=...).
        Note: in statsmodels ARIMA supports seasonal_order in recent versions.
    trend : str or None
        Trend term passed to statsmodels ('n','c','t','ct') or None.
    enforce_stationarity, enforce_invertibility : bool
        Passed through to statsmodels.

    Returns
    -------
    ArimaTrendForecast
        Contains forecast array and information criteria when available.
    '''
    _require_statsmodels()
    from statsmodels.tsa.arima.model import ARIMA

    y = np.asarray(t_hat_train, dtype=float).ravel()
    if h <= 0:
        return ArimaTrendForecast(forecast=np.array([], dtype=float), order=tuple(order))

    if y.size < 5:
        raise ValueError("Need at least 5 points in t_hat_train to fit ARIMA reliably.")

    kwargs = dict(
        order=tuple(order),
        trend=trend,
        enforce_stationarity=enforce_stationarity,
        enforce_invertibility=enforce_invertibility,
    )
    if seasonal_order is not None:
        kwargs["seasonal_order"] = tuple(seasonal_order)

    model = ARIMA(y, **kwargs)
    res = model.fit()

    fc = res.forecast(steps=int(h))
    # statsmodels returns pandas or array-like; normalize
    fc = np.asarray(fc, dtype=float).ravel()

    aic = getattr(res, "aic", None)
    bic = getattr(res, "bic", None)

    return ArimaTrendForecast(
        forecast=fc,
        order=tuple(order),
        aic=float(aic) if aic is not None else None,
        bic=float(bic) if bic is not None else None,
    )
