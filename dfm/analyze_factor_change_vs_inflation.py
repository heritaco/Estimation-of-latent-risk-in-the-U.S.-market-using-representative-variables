from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, grangercausalitytests


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
ANALYSIS_DIR = OUTPUT_DIR / "inflation_change_analysis"

FACTOR_PATH = OUTPUT_DIR / "factor_latente_final.csv"
INFLATION_PATH = DATA_DIR / "us_monthly_inflation_rates_1913_2026.csv"
ANALYSIS_START = "2004-01-01"
MAX_GRANGER_LAG = 12
MAX_CROSS_CORR_LAG = 12


def log(message: str) -> None:
    print(message, flush=True)


def ensure_dirs() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def load_factor(path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"Missing factor file: {path}")
    frame = pd.read_csv(path, parse_dates=["Date"])
    frame = frame.sort_values("Date").set_index("Date")
    if frame.empty:
        raise ValueError("Latent factor file is empty")
    series = pd.to_numeric(frame.iloc[:, 0], errors="coerce").dropna()
    series.name = "latent_factor_daily"
    return series


def load_monthly_inflation(path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"Missing inflation file: {path}")

    frame = pd.read_csv(path)
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    expected = {"Year", *month_order}
    if not expected.issubset(frame.columns):
        raise ValueError(f"Unexpected monthly inflation format in {path}. Columns={list(frame.columns)}")

    melted = frame.melt(id_vars=["Year"], value_vars=month_order, var_name="month", value_name="inflation_rate_pct")
    melted["Year"] = pd.to_numeric(melted["Year"], errors="coerce")
    melted["month_num"] = pd.to_datetime(melted["month"], format="%b").dt.month
    melted["Date"] = pd.to_datetime(
        {"year": melted["Year"], "month": melted["month_num"], "day": 1},
        errors="coerce",
    )

    series = pd.Series(
        pd.to_numeric(melted["inflation_rate_pct"], errors="coerce").to_numpy(),
        index=melted["Date"],
        name="inflation_rate_pct",
    )
    series = series[~series.index.isna()].dropna().sort_index()
    if series.empty:
        raise ValueError(f"Parsed monthly inflation series is empty: {path}")
    return series


def symmetric_pct_change(series: pd.Series) -> pd.Series:
    previous = series.shift(1)
    denominator = previous.abs() + series.abs()
    result = 200.0 * (series - previous) / denominator
    result = result.replace([np.inf, -np.inf], np.nan)
    return result.rename("factor_symmetric_pct_change")


def build_monthly_change_frame(factor_daily: pd.Series, inflation: pd.Series) -> pd.DataFrame:
    factor_daily = factor_daily[factor_daily.index >= pd.Timestamp(ANALYSIS_START)]
    inflation = inflation[inflation.index >= pd.Timestamp(ANALYSIS_START)]

    factor_monthly_mean = factor_daily.resample("ME").mean().rename("latent_factor_monthly_mean")
    factor_monthly_last = factor_daily.resample("ME").last().rename("latent_factor_monthly_last")
    factor_change = symmetric_pct_change(factor_monthly_mean)

    inflation.index = inflation.index.to_period("M").to_timestamp(how="end").normalize()
    inflation = inflation[~inflation.index.duplicated(keep="last")].sort_index()

    merged = pd.concat(
        [
            factor_monthly_mean,
            factor_monthly_last,
            factor_change,
            inflation.rename("inflation_rate_pct"),
        ],
        axis=1,
        sort=True,
    ).dropna()
    if merged.empty:
        raise ValueError("No overlap between factor monthly change and monthly inflation after alignment")
    return merged


def compute_basic_correlations(frame: pd.DataFrame) -> Dict[str, float]:
    factor_change = frame["factor_symmetric_pct_change"]
    inflation = frame["inflation_rate_pct"]
    return {
        "pearson_contemporaneous": float(factor_change.corr(inflation, method="pearson")),
        "spearman_contemporaneous": float(factor_change.corr(inflation, method="spearman")),
    }


def compute_cross_correlation_table(frame: pd.DataFrame, max_lag: int = MAX_CROSS_CORR_LAG) -> pd.DataFrame:
    factor_change = frame["factor_symmetric_pct_change"]
    inflation = frame["inflation_rate_pct"]
    rows: List[Dict[str, float]] = []

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            aligned = pd.concat(
                [factor_change.rename("factor_change"), inflation.shift(-lag).rename("inflation_shifted")],
                axis=1,
            ).dropna()
        else:
            aligned = pd.concat(
                [factor_change.shift(lag).rename("factor_change_shifted"), inflation.rename("inflation")],
                axis=1,
            ).dropna()

        correlation = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1])) if len(aligned) >= 3 else np.nan
        rows.append({"lag_periods": lag, "correlation": correlation, "n_obs": int(len(aligned))})

    return pd.DataFrame(rows)


def adf_summary(series: pd.Series) -> Dict[str, float]:
    clean = series.dropna()
    result = adfuller(clean, autolag="AIC")
    return {
        "adf_stat": float(result[0]),
        "p_value": float(result[1]),
        "used_lag": int(result[2]),
        "n_obs": int(result[3]),
    }


def granger_summary(frame: pd.DataFrame, cause_col: str, effect_col: str, max_lag: int = MAX_GRANGER_LAG) -> Dict[str, object]:
    data = frame[[effect_col, cause_col]].dropna()
    if len(data) < max_lag + 10:
        raise ValueError(f"Not enough observations for Granger test {cause_col} -> {effect_col}")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")
        results = grangercausalitytests(data, maxlag=max_lag, verbose=False)

    lag_rows = []
    for lag, payload in results.items():
        test_stats = payload[0]["ssr_ftest"]
        lag_rows.append(
            {
                "lag": int(lag),
                "f_stat": float(test_stats[0]),
                "p_value": float(test_stats[1]),
            }
        )

    lag_frame = pd.DataFrame(lag_rows).sort_values("lag")
    best_row = lag_frame.loc[lag_frame["p_value"].idxmin()]
    significant_lags = lag_frame.loc[lag_frame["p_value"] < 0.05, "lag"].astype(int).tolist()

    return {
        "direction": f"{cause_col} -> {effect_col}",
        "max_lag": int(max_lag),
        "significant_at_5pct": bool(len(significant_lags) > 0),
        "significant_lags": significant_lags,
        "best_lag": int(best_row["lag"]),
        "best_p_value": float(best_row["p_value"]),
        "best_f_stat": float(best_row["f_stat"]),
        "lag_results": lag_frame.to_dict(orient="records"),
    }


def write_json(payload: Dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    log(f"[plot] saved {path.relative_to(ROOT_DIR).as_posix()}")


def plot_change_vs_inflation(frame: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()

    ax1.plot(
        frame.index,
        frame["factor_symmetric_pct_change"],
        color="#1d4ed8",
        linewidth=1.6,
        label="Factor symmetric % change",
    )
    ax2.plot(
        frame.index,
        frame["inflation_rate_pct"],
        color="#b91c1c",
        linewidth=1.6,
        label="Monthly inflation rate",
    )

    ax1.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax2.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax1.set_title("Factor monthly change vs monthly inflation")
    ax1.set_ylabel("Factor symmetric % change")
    ax2.set_ylabel("Inflation rate (%)")
    ax1.grid(alpha=0.25)

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left")
    save_figure(ANALYSIS_DIR / "factor_change_vs_inflation_timeseries.png")


def plot_scatter(frame: pd.DataFrame, corr_stats: Dict[str, float]) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        frame["factor_symmetric_pct_change"],
        frame["inflation_rate_pct"],
        alpha=0.65,
        color="#0f766e",
        edgecolor="none",
    )
    ax.set_title("Factor monthly change vs inflation")
    ax.set_xlabel("Factor symmetric % change")
    ax.set_ylabel("Inflation rate (%)")
    ax.grid(alpha=0.25)
    ax.text(
        0.03,
        0.97,
        (
            f"Pearson: {corr_stats['pearson_contemporaneous']:.4f}\n"
            f"Spearman: {corr_stats['spearman_contemporaneous']:.4f}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )
    save_figure(ANALYSIS_DIR / "factor_change_vs_inflation_scatter.png")


def plot_cross_correlation(cross_corr: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(cross_corr["lag_periods"], cross_corr["correlation"], color="#7c3aed", alpha=0.85)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Cross-correlation: factor monthly change vs inflation")
    ax.set_xlabel("Lag in months")
    ax.set_ylabel("Correlation")
    ax.grid(axis="y", alpha=0.25)
    save_figure(ANALYSIS_DIR / "factor_change_inflation_cross_correlation.png")


def write_summary(
    path: Path,
    frame: pd.DataFrame,
    corr_stats: Dict[str, float],
    cross_corr: pd.DataFrame,
    adf_factor_change: Dict[str, float],
    adf_inflation: Dict[str, float],
    factor_to_inflation: Dict[str, object],
    inflation_to_factor: Dict[str, object],
) -> None:
    best_cross = cross_corr.loc[cross_corr["correlation"].abs().idxmax()]

    lines = [
        "Factor change vs U.S. monthly inflation analysis",
        "=" * 47,
        "",
        f"Sample start: {frame.index.min().date()}",
        f"Sample end: {frame.index.max().date()}",
        f"Monthly observations: {len(frame)}",
        "",
        "Factor transformation:",
        "Daily latent factor aggregated to monthly mean, then transformed using symmetric percentage change:",
        "200 * (x_t - x_{t-1}) / (|x_t| + |x_{t-1}|).",
        "This is used instead of simple pct_change because the latent factor can cross zero.",
        "",
        "Inflation source:",
        f"{INFLATION_PATH.relative_to(ROOT_DIR).as_posix()}",
        "",
        "Contemporaneous correlation:",
        f"Pearson: {corr_stats['pearson_contemporaneous']:.6f}",
        f"Spearman: {corr_stats['spearman_contemporaneous']:.6f}",
        "",
        f"Strongest cross-correlation in +/- {MAX_CROSS_CORR_LAG} months:",
        f"Lag: {int(best_cross['lag_periods'])} months",
        f"Correlation: {float(best_cross['correlation']):.6f}",
        "",
        "ADF stationarity checks:",
        f"Factor symmetric % change p-value: {adf_factor_change['p_value']:.6f}",
        f"Inflation rate p-value: {adf_inflation['p_value']:.6f}",
        "",
        "Granger causality note:",
        "This is predictive causality, not structural economic causation.",
        "",
        "Factor change -> Inflation:",
        f"Significant at 5%: {factor_to_inflation['significant_at_5pct']}",
        f"Best lag: {factor_to_inflation['best_lag']}",
        f"Best p-value: {factor_to_inflation['best_p_value']:.6f}",
        f"Significant lags: {factor_to_inflation['significant_lags']}",
        "",
        "Inflation -> Factor change:",
        f"Significant at 5%: {inflation_to_factor['significant_at_5pct']}",
        f"Best lag: {inflation_to_factor['best_lag']}",
        f"Best p-value: {inflation_to_factor['best_p_value']:.6f}",
        f"Significant lags: {inflation_to_factor['significant_lags']}",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    log(f"[output] wrote {path.relative_to(ROOT_DIR).as_posix()}")


def main() -> None:
    ensure_dirs()
    log("[start] factor percentage-change vs monthly inflation analysis")

    factor_daily = load_factor(FACTOR_PATH)
    log(f"[data] loaded latent factor with {len(factor_daily)} daily observations")

    inflation = load_monthly_inflation(INFLATION_PATH)
    log(f"[data] loaded monthly inflation with {len(inflation)} observations from {INFLATION_PATH.relative_to(ROOT_DIR).as_posix()}")

    merged = build_monthly_change_frame(factor_daily, inflation)
    merged.to_csv(ANALYSIS_DIR / "factor_change_vs_inflation_merged_monthly.csv", index_label="Date")
    log(f"[output] wrote {(ANALYSIS_DIR / 'factor_change_vs_inflation_merged_monthly.csv').relative_to(ROOT_DIR).as_posix()}")

    corr_stats = compute_basic_correlations(merged)
    cross_corr = compute_cross_correlation_table(merged)
    cross_corr.to_csv(ANALYSIS_DIR / "factor_change_vs_inflation_cross_correlation.csv", index=False)
    log(f"[output] wrote {(ANALYSIS_DIR / 'factor_change_vs_inflation_cross_correlation.csv').relative_to(ROOT_DIR).as_posix()}")

    adf_factor_change = adf_summary(merged["factor_symmetric_pct_change"])
    adf_inflation = adf_summary(merged["inflation_rate_pct"])

    factor_to_inflation = granger_summary(merged, "factor_symmetric_pct_change", "inflation_rate_pct")
    inflation_to_factor = granger_summary(merged, "inflation_rate_pct", "factor_symmetric_pct_change")

    stats_payload = {
        "sample_start": str(merged.index.min().date()),
        "sample_end": str(merged.index.max().date()),
        "n_months": int(len(merged)),
        "factor_transform": "monthly mean -> symmetric percentage change",
        "inflation_source": str(INFLATION_PATH.relative_to(ROOT_DIR).as_posix()),
        "correlations": corr_stats,
        "adf_factor_symmetric_pct_change": adf_factor_change,
        "adf_inflation_rate": adf_inflation,
        "granger_factor_change_to_inflation": factor_to_inflation,
        "granger_inflation_to_factor_change": inflation_to_factor,
    }
    write_json(stats_payload, ANALYSIS_DIR / "factor_change_vs_inflation_stats.json")
    log(f"[output] wrote {(ANALYSIS_DIR / 'factor_change_vs_inflation_stats.json').relative_to(ROOT_DIR).as_posix()}")

    write_summary(
        ANALYSIS_DIR / "factor_change_vs_inflation_summary.txt",
        merged,
        corr_stats,
        cross_corr,
        adf_factor_change,
        adf_inflation,
        factor_to_inflation,
        inflation_to_factor,
    )

    plot_change_vs_inflation(merged)
    plot_scatter(merged, corr_stats)
    plot_cross_correlation(cross_corr)

    log("[done] factor change vs inflation analysis finished")


if __name__ == "__main__":
    main()
