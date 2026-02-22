# -*- coding: utf-8 -*-
'''Convenience loaders: arrays, pandas, CSV, optional Yahoo Finance.'''
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SeriesMeta:
    name: str
    start: Optional[str] = None
    end: Optional[str] = None
    use_log: bool = True
    freq: Optional[str] = None


def from_array(Z: np.ndarray, name: str = "series", use_log: bool = False) -> Tuple[np.ndarray, SeriesMeta]:
    Z = np.asarray(Z, dtype=float).ravel()
    if use_log:
        Z = np.log(Z)
    return Z, SeriesMeta(name=name, use_log=use_log)


def from_pandas(series: "pd.Series", name: Optional[str] = None, use_log: bool = False) -> Tuple[np.ndarray, SeriesMeta]:
    s = series.dropna()
    Z = s.to_numpy(dtype=float).ravel()
    if use_log:
        Z = np.log(Z)
    nm = name or (getattr(series, "name", None) or "series")
    meta = SeriesMeta(
        name=str(nm),
        start=str(s.index[0]) if hasattr(s.index, "__len__") and len(s.index) else None,
        end=str(s.index[-1]) if hasattr(s.index, "__len__") and len(s.index) else None,
        use_log=use_log,
    )
    return Z, meta


def from_csv(
    csv_path: str,
    date_col: Optional[str] = "Date",
    value_col: str = "Adj Close",
    use_log: bool = True,
) -> Tuple[np.ndarray, SeriesMeta]:
    df = pd.read_csv(csv_path)
    if date_col is not None and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col)
        idx = df[date_col]
    else:
        idx = None

    if value_col not in df.columns:
        raise ValueError(f"value_col='{value_col}' not found in CSV columns={list(df.columns)}")

    s = df[value_col].astype(float)
    Z = s.to_numpy(dtype=float).ravel()
    if use_log:
        Z = np.log(Z)

    meta = SeriesMeta(
        name=f"csv:{value_col}",
        start=str(idx.iloc[0].date()) if idx is not None else None,
        end=str(idx.iloc[-1].date()) if idx is not None else None,
        use_log=use_log,
    )
    return Z, meta


def from_yahoo(
    ticker: str,
    start: str = "2010-01-01",
    end: Optional[str] = None,
    use_log: bool = True,
) -> Tuple[np.ndarray, SeriesMeta]:
    try:
        import yfinance as yf
    except Exception as e:
        raise ImportError("Install optional extra: `pip install guerrero-trend[yahoo]`.") from e

    df = yf.download(ticker, start=start, end=end)
    if df.empty:
        raise RuntimeError("No data downloaded (check ticker/start/end).")

    col = None
    for c in ["Adj Close", "AdjClose", "Close", "close"]:
        if c in df.columns:
            col = c
            break
    if col is None:
        raise ValueError("No price column found (Adj Close/Close).")

    s = df[col].dropna()
    Z = s.to_numpy(dtype=float).ravel()
    if use_log:
        Z = np.log(Z)

    meta = SeriesMeta(
        name=ticker,
        start=str(s.index[0].date()),
        end=str(s.index[-1].date()),
        use_log=use_log,
    )
    return Z, meta
