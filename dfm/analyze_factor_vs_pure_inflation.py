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
ANALYSIS_DIR = OUTPUT_DIR / "pure_inflation_analysis"

FACTOR_PATH = OUTPUT_DIR / "factor_latente_final.csv"
INFLATION_PATH = DATA_DIR / "us_monthly_inflation_rates_1913_2026.csv"
ANALYSIS_START = "2004-01-01"
MAX_GRANGER_LAG = 12
MAX_CROSS_CORR_LAG = 12
FULL_INDEX_BASE = 100.0
SAMPLE_INDEX_BASE = 100.0


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


def load_monthly_inflation_rates(path: Path) -> pd.Series:
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


def build_full_price_index(inflation_rate: pd.Series, base: float = FULL_INDEX_BASE) -> pd.Series:
    gross_rate = 1.0 + inflation_rate / 100.0
    if (gross_rate <= 0).any():
        raise ValueError("Inflation rates imply non-positive gross rates; cannot build cumulative price index")
    price_index = base * gross_rate.cumprod()
    price_index.name = "price_index_full"
    return price_index


def build_monthly_level_frame(factor_daily: pd.Series, inflation_rate: pd.Series) -> pd.DataFrame:
    factor_daily = factor_daily[factor_daily.index >= pd.Timestamp(ANALYSIS_START)]

    inflation_rate = inflation_rate.sort_index()
    inflation_rate_period_end = pd.Series(
        inflation_rate.to_numpy(),
        index=inflation_rate.index.to_period("M").to_timestamp(how="end").normalize(),
        name="inflation_rate_pct",
    )
    inflation_rate_period_end = inflation_rate_period_end[~inflation_rate_period_end.index.duplicated(keep="last")].sort_index()

    full_index = build_full_price_index(inflation_rate_period_end, base=FULL_INDEX_BASE)
    inflation_rate_period_end = inflation_rate_period_end[inflation_rate_period_end.index >= pd.Timestamp(ANALYSIS_START)]
    full_index = full_index[full_index.index >= pd.Timestamp(ANALYSIS_START)]

    factor_monthly_mean = factor_daily.resample("ME").mean().rename("latent_factor_monthly_mean")
    factor_monthly_last = factor_daily.resample("ME").last().rename("latent_factor_monthly_last")

    merged = pd.concat(
        [
            factor_monthly_mean,
            factor_monthly_last,
            inflation_rate_period_end.rename("inflation_rate_pct"),
            full_index.rename("price_index_full"),
        ],
        axis=1,
        sort=True,
    ).dropna()
    if merged.empty:
        raise ValueError("No overlap between latent factor and pure inflation index after monthly alignment")

    merged["price_index_rebased"] = SAMPLE_INDEX_BASE * merged["price_index_full"] / float(merged["price_index_full"].iloc[0])
    return merged


def compute_basic_correlations(frame: pd.DataFrame) -> Dict[str, float]:
    factor = frame["latent_factor_monthly_mean"]
    pure_inflation = frame["price_index_rebased"]
    return {
        "pearson_contemporaneous": float(factor.corr(pure_inflation, method="pearson")),
        "spearman_contemporaneous": float(factor.corr(pure_inflation, method="spearman")),
    }


def compute_cross_correlation_table(frame: pd.DataFrame, max_lag: int = MAX_CROSS_CORR_LAG) -> pd.DataFrame:
    factor = frame["latent_factor_monthly_mean"]
    pure_inflation = frame["price_index_rebased"]
    rows: List[Dict[str, float]] = []

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            aligned = pd.concat(
                [factor.rename("factor"), pure_inflation.shift(-lag).rename("pure_inflation_shifted")],
                axis=1,
            ).dropna()
        else:
            aligned = pd.concat(
                [factor.shift(lag).rename("factor_shifted"), pure_inflation.rename("pure_inflation")],
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


def plot_factor_vs_pure_inflation(frame: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()

    ax1.plot(
        frame.index,
        frame["latent_factor_monthly_mean"],
        color="#1d4ed8",
        linewidth=1.7,
        label="Latent factor (monthly mean)",
    )
    ax2.plot(
        frame.index,
        frame["price_index_rebased"],
        color="#b91c1c",
        linewidth=1.9,
        label="Pure inflation index (rebased)",
    )

    ax1.set_title("Latent factor vs pure inflation")
    ax1.set_ylabel("Latent factor")
    ax2.set_ylabel("Inflation index (base 100)")
    ax1.grid(alpha=0.25)

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left")
    save_figure(ANALYSIS_DIR / "factor_vs_pure_inflation_timeseries.png")


def plot_scatter(frame: pd.DataFrame, corr_stats: Dict[str, float]) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        frame["latent_factor_monthly_mean"],
        frame["price_index_rebased"],
        alpha=0.65,
        color="#0f766e",
        edgecolor="none",
    )
    ax.set_title("Latent factor vs pure inflation")
    ax.set_xlabel("Latent factor (monthly mean)")
    ax.set_ylabel("Inflation index (base 100)")
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
    save_figure(ANALYSIS_DIR / "factor_vs_pure_inflation_scatter.png")


def plot_cross_correlation(cross_corr: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(cross_corr["lag_periods"], cross_corr["correlation"], color="#7c3aed", alpha=0.85)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Cross-correlation: factor vs pure inflation")
    ax.set_xlabel("Lag in months")
    ax.set_ylabel("Correlation")
    ax.grid(axis="y", alpha=0.25)
    save_figure(ANALYSIS_DIR / "factor_pure_inflation_cross_correlation.png")


def write_summary(
    path: Path,
    frame: pd.DataFrame,
    corr_stats: Dict[str, float],
    cross_corr: pd.DataFrame,
    adf_factor: Dict[str, float],
    adf_pure_inflation: Dict[str, float],
    factor_to_inflation: Dict[str, object],
    inflation_to_factor: Dict[str, object],
) -> None:
    best_cross = cross_corr.loc[cross_corr["correlation"].abs().idxmax()]

    lines = [
        "Factor vs pure inflation analysis",
        "=" * 34,
        "",
        f"Sample start: {frame.index.min().date()}",
        f"Sample end: {frame.index.max().date()}",
        f"Monthly observations: {len(frame)}",
        "",
        "Pure inflation construction:",
        f"Source monthly inflation rates: {INFLATION_PATH.relative_to(ROOT_DIR).as_posix()}",
        "Pure inflation index = cumulative product of (1 + monthly_rate/100).",
        f"Full-history base: {FULL_INDEX_BASE}",
        f"Sample-rebased level for analysis: {SAMPLE_INDEX_BASE} at first overlap month.",
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
        f"Latent factor monthly mean p-value: {adf_factor['p_value']:.6f}",
        f"Pure inflation index p-value: {adf_pure_inflation['p_value']:.6f}",
        "",
        "Granger causality note:",
        "This is predictive causality, not structural economic causation.",
        "Because pure inflation is a level series, non-stationarity can matter for interpretation.",
        "",
        "Factor -> Pure inflation:",
        f"Significant at 5%: {factor_to_inflation['significant_at_5pct']}",
        f"Best lag: {factor_to_inflation['best_lag']}",
        f"Best p-value: {factor_to_inflation['best_p_value']:.6f}",
        f"Significant lags: {factor_to_inflation['significant_lags']}",
        "",
        "Pure inflation -> Factor:",
        f"Significant at 5%: {inflation_to_factor['significant_at_5pct']}",
        f"Best lag: {inflation_to_factor['best_lag']}",
        f"Best p-value: {inflation_to_factor['best_p_value']:.6f}",
        f"Significant lags: {inflation_to_factor['significant_lags']}",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    log(f"[output] wrote {path.relative_to(ROOT_DIR).as_posix()}")


def main() -> None:
    ensure_dirs()
    log("[start] factor vs pure inflation analysis")

    factor_daily = load_factor(FACTOR_PATH)
    log(f"[data] loaded latent factor with {len(factor_daily)} daily observations")

    inflation_rate = load_monthly_inflation_rates(INFLATION_PATH)
    log(f"[data] loaded monthly inflation rates with {len(inflation_rate)} observations")

    merged = build_monthly_level_frame(factor_daily, inflation_rate)
    merged.to_csv(ANALYSIS_DIR / "factor_vs_pure_inflation_merged_monthly.csv", index_label="Date")
    log(f"[output] wrote {(ANALYSIS_DIR / 'factor_vs_pure_inflation_merged_monthly.csv').relative_to(ROOT_DIR).as_posix()}")

    corr_stats = compute_basic_correlations(merged)
    cross_corr = compute_cross_correlation_table(merged)
    cross_corr.to_csv(ANALYSIS_DIR / "factor_vs_pure_inflation_cross_correlation.csv", index=False)
    log(f"[output] wrote {(ANALYSIS_DIR / 'factor_vs_pure_inflation_cross_correlation.csv').relative_to(ROOT_DIR).as_posix()}")

    adf_factor = adf_summary(merged["latent_factor_monthly_mean"])
    adf_pure_inflation = adf_summary(merged["price_index_rebased"])

    factor_to_inflation = granger_summary(merged, "latent_factor_monthly_mean", "price_index_rebased")
    inflation_to_factor = granger_summary(merged, "price_index_rebased", "latent_factor_monthly_mean")

    stats_payload = {
        "sample_start": str(merged.index.min().date()),
        "sample_end": str(merged.index.max().date()),
        "n_months": int(len(merged)),
        "inflation_source": str(INFLATION_PATH.relative_to(ROOT_DIR).as_posix()),
        "pure_inflation_construction": {
            "full_history_base": FULL_INDEX_BASE,
            "sample_rebased_base": SAMPLE_INDEX_BASE,
            "formula": "cumprod(1 + inflation_rate/100)",
        },
        "correlations": corr_stats,
        "adf_factor_monthly_mean": adf_factor,
        "adf_pure_inflation_index": adf_pure_inflation,
        "granger_factor_to_pure_inflation": factor_to_inflation,
        "granger_pure_inflation_to_factor": inflation_to_factor,
    }
    write_json(stats_payload, ANALYSIS_DIR / "factor_vs_pure_inflation_stats.json")
    log(f"[output] wrote {(ANALYSIS_DIR / 'factor_vs_pure_inflation_stats.json').relative_to(ROOT_DIR).as_posix()}")

    write_summary(
        ANALYSIS_DIR / "factor_vs_pure_inflation_summary.txt",
        merged,
        corr_stats,
        cross_corr,
        adf_factor,
        adf_pure_inflation,
        factor_to_inflation,
        inflation_to_factor,
    )

    plot_factor_vs_pure_inflation(merged)
    plot_scatter(merged, corr_stats)
    plot_cross_correlation(cross_corr)

    log("[done] factor vs pure inflation analysis finished")


if __name__ == "__main__":
    main()
