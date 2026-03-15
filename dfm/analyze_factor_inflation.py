from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, grangercausalitytests


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
ANALYSIS_DIR = OUTPUT_DIR / "inflation_analysis"

FACTOR_PATH = OUTPUT_DIR / "factor_latente_final.csv"
INFLATION_CANDIDATE_PATHS = [
    DATA_DIR / "FPCPITOYLZGUSA.csv",
    DATA_DIR / "FPCPITOTLZGUSA.csv",
]
ANALYSIS_START = "2004-01-01"
MAX_GRANGER_LAG = 12
MAX_CROSS_CORR_LAG = 12


def log(message: str) -> None:
    print(message, flush=True)


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
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


def resolve_inflation_path() -> Path:
    for path in INFLATION_CANDIDATE_PATHS:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No local inflation CSV was found. Expected one of: "
        + ", ".join(path.name for path in INFLATION_CANDIDATE_PATHS)
    )


def infer_periodicity(index: pd.DatetimeIndex) -> Tuple[str, str, int]:
    if len(index) < 2:
        return "YE", "annual", 4

    deltas = pd.Series(index.sort_values()).diff().dropna().dt.days
    median_days = float(deltas.median())
    if median_days <= 120:
        return "QE", "quarterly", 8
    if median_days <= 400:
        return "YE", "annual", 4
    return "YE", "annual", 4


def period_alias_from_resample_rule(resample_rule: str) -> str:
    if resample_rule.startswith("Q"):
        return "Q"
    if resample_rule.startswith("Y"):
        return "Y"
    raise ValueError(f"Unsupported resample rule: {resample_rule}")


def align_index_to_period_end(series: pd.Series, resample_rule: str) -> pd.Series:
    alias = period_alias_from_resample_rule(resample_rule)
    aligned_index = series.index.to_period(alias).to_timestamp(how="end").normalize()
    aligned = pd.Series(series.to_numpy(), index=aligned_index, name=series.name)
    aligned = aligned[~aligned.index.duplicated(keep="last")].sort_index()
    return aligned


def load_local_inflation() -> Tuple[pd.Series, Dict[str, object]]:
    path = resolve_inflation_path()
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Inflation CSV is empty: {path}")

    lower_map = {str(column).lower(): column for column in frame.columns}
    date_col = lower_map.get("observation_date") or lower_map.get("date") or frame.columns[0]
    value_candidates = [column for column in frame.columns if column != date_col]
    if not value_candidates:
        raise ValueError(f"Inflation CSV has no value column: {path}")
    value_col = value_candidates[0]

    index = pd.to_datetime(frame[date_col], errors="coerce")
    values = pd.to_numeric(frame[value_col], errors="coerce").to_numpy()
    series = pd.Series(values, index=index, name="inflation_rate_pct")
    series = series[~series.index.isna()].dropna().sort_index()
    if series.empty:
        raise ValueError(f"Parsed inflation series is empty: {path}")

    resample_rule, frequency_label, suggested_lag = infer_periodicity(series.index)
    metadata = {
        "path": str(path.relative_to(ROOT_DIR).as_posix()),
        "value_column": value_col,
        "date_column": date_col,
        "frequency_label": frequency_label,
        "resample_rule": resample_rule,
        "suggested_granger_lag": suggested_lag,
    }
    return series, metadata


def build_period_analysis_frame(
    factor_daily: pd.Series,
    inflation_rate: pd.Series,
    resample_rule: str,
) -> pd.DataFrame:
    factor_daily = factor_daily[factor_daily.index >= pd.Timestamp(ANALYSIS_START)]
    inflation_rate = inflation_rate[inflation_rate.index >= pd.Timestamp(ANALYSIS_START)]
    inflation_rate = align_index_to_period_end(inflation_rate, resample_rule)

    factor_period_mean = factor_daily.resample(resample_rule).mean().rename("latent_factor_period_mean")
    factor_period_last = factor_daily.resample(resample_rule).last().rename("latent_factor_period_last")

    merged = pd.concat(
        [factor_period_mean, factor_period_last, inflation_rate.rename("inflation_rate_pct")],
        axis=1,
        sort=True,
    )
    merged = merged.dropna()
    if merged.empty:
        raise ValueError("No overlap between latent factor and local inflation series after period alignment")
    return merged


def compute_basic_correlations(frame: pd.DataFrame) -> Dict[str, float]:
    factor = frame["latent_factor_period_mean"]
    inflation = frame["inflation_rate_pct"]
    return {
        "pearson_contemporaneous": float(factor.corr(inflation, method="pearson")),
        "spearman_contemporaneous": float(factor.corr(inflation, method="spearman")),
    }


def compute_cross_correlation_table(frame: pd.DataFrame, max_lag: int = MAX_CROSS_CORR_LAG) -> pd.DataFrame:
    factor = frame["latent_factor_period_mean"]
    inflation = frame["inflation_rate_pct"]
    rows: List[Dict[str, float]] = []

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            aligned = pd.concat(
                [
                    factor.rename("factor"),
                    inflation.shift(-lag).rename("inflation_shifted"),
                ],
                axis=1,
            ).dropna()
        else:
            aligned = pd.concat(
                [
                    factor.shift(lag).rename("factor_shifted"),
                    inflation.rename("inflation"),
                ],
                axis=1,
            ).dropna()

        correlation = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1])) if len(aligned) >= 3 else np.nan
        rows.append(
            {
                "lag_periods": lag,
                "correlation": correlation,
                "n_obs": int(len(aligned)),
            }
        )

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


def plot_timeseries(frame: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()

    ax1.plot(
        frame.index,
        frame["latent_factor_period_mean"],
        color="#1d4ed8",
        linewidth=1.8,
        label="Latent factor (period mean)",
    )
    ax2.plot(
        frame.index,
        frame["inflation_rate_pct"],
        color="#b91c1c",
        linewidth=1.8,
        label="Inflation",
    )

    ax1.set_title("Latent factor vs U.S. inflation")
    ax1.set_ylabel("Latent factor")
    ax2.set_ylabel("Inflation (%)")
    ax1.grid(alpha=0.25)

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left")
    save_figure(ANALYSIS_DIR / "factor_vs_inflation_timeseries.png")


def plot_scatter(frame: pd.DataFrame, corr_stats: Dict[str, float]) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        frame["latent_factor_period_mean"],
        frame["inflation_rate_pct"],
        alpha=0.65,
        color="#0f766e",
        edgecolor="none",
    )
    ax.set_title("Latent factor vs inflation")
    ax.set_xlabel("Latent factor (period mean)")
    ax.set_ylabel("Inflation (%)")
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
    save_figure(ANALYSIS_DIR / "factor_vs_inflation_scatter.png")


def plot_cross_correlation(cross_corr: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(cross_corr["lag_periods"], cross_corr["correlation"], color="#7c3aed", alpha=0.85)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Cross-correlation: factor vs inflation")
    ax.set_xlabel("Lag in periods")
    ax.set_ylabel("Correlation")
    ax.grid(axis="y", alpha=0.25)
    save_figure(ANALYSIS_DIR / "factor_inflation_cross_correlation.png")


def write_summary(
    path: Path,
    frame: pd.DataFrame,
    inflation_meta: Dict[str, object],
    corr_stats: Dict[str, float],
    cross_corr: pd.DataFrame,
    adf_factor: Dict[str, float],
    adf_inflation: Dict[str, float],
    factor_to_inflation: Dict[str, object],
    inflation_to_factor: Dict[str, object],
) -> None:
    best_cross = cross_corr.loc[cross_corr["correlation"].abs().idxmax()]

    lines = [
        "Factor vs U.S. inflation analysis",
        "=" * 36,
        "",
        f"Sample start: {frame.index.min().date()}",
        f"Sample end: {frame.index.max().date()}",
        f"{inflation_meta['frequency_label'].capitalize()} observations: {len(frame)}",
        "",
        "Inflation definition:",
        f"Local file: {inflation_meta['path']}",
        f"Column used: {inflation_meta['value_column']}",
        f"Detected frequency: {inflation_meta['frequency_label']}",
        "The daily latent factor was aggregated to the same frequency using the mean over each period.",
        "",
        "Contemporaneous correlation:",
        f"Pearson: {corr_stats['pearson_contemporaneous']:.6f}",
        f"Spearman: {corr_stats['spearman_contemporaneous']:.6f}",
        "",
        f"Strongest cross-correlation in +/- {MAX_CROSS_CORR_LAG} periods:",
        f"Lag: {int(best_cross['lag_periods'])} periods",
        f"Correlation: {float(best_cross['correlation']):.6f}",
        "",
        "ADF stationarity checks:",
        f"Latent factor period mean p-value: {adf_factor['p_value']:.6f}",
        f"Inflation p-value: {adf_inflation['p_value']:.6f}",
        "",
        "Granger causality note:",
        "This is predictive causality, not structural economic causation.",
        "",
        "Factor -> Inflation:",
        f"Significant at 5%: {factor_to_inflation['significant_at_5pct']}",
        f"Best lag: {factor_to_inflation['best_lag']}",
        f"Best p-value: {factor_to_inflation['best_p_value']:.6f}",
        f"Significant lags: {factor_to_inflation['significant_lags']}",
        "",
        "Inflation -> Factor:",
        f"Significant at 5%: {inflation_to_factor['significant_at_5pct']}",
        f"Best lag: {inflation_to_factor['best_lag']}",
        f"Best p-value: {inflation_to_factor['best_p_value']:.6f}",
        f"Significant lags: {inflation_to_factor['significant_lags']}",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    log(f"[output] wrote {path.relative_to(ROOT_DIR).as_posix()}")


def main() -> None:
    ensure_dirs()
    log("[start] factor vs U.S. inflation analysis")

    factor_daily = load_factor(FACTOR_PATH)
    log(f"[data] loaded latent factor with {len(factor_daily)} daily observations")

    inflation_series, inflation_meta = load_local_inflation()
    log(
        f"[data] loaded local inflation series with {len(inflation_series)} observations "
        f"from {inflation_meta['path']} ({inflation_meta['frequency_label']})"
    )

    merged = build_period_analysis_frame(factor_daily, inflation_series, str(inflation_meta["resample_rule"]))
    merged.to_csv(ANALYSIS_DIR / "factor_inflation_merged_periodic.csv", index_label="Date")
    log(f"[output] wrote {(ANALYSIS_DIR / 'factor_inflation_merged_periodic.csv').relative_to(ROOT_DIR).as_posix()}")

    corr_stats = compute_basic_correlations(merged)
    cross_corr = compute_cross_correlation_table(merged)
    cross_corr.to_csv(ANALYSIS_DIR / "factor_inflation_cross_correlation.csv", index=False)
    log(f"[output] wrote {(ANALYSIS_DIR / 'factor_inflation_cross_correlation.csv').relative_to(ROOT_DIR).as_posix()}")

    adf_factor = adf_summary(merged["latent_factor_period_mean"])
    adf_inflation = adf_summary(merged["inflation_rate_pct"])

    max_lag = min(int(inflation_meta["suggested_granger_lag"]), max(1, len(merged) // 5))
    factor_to_inflation = granger_summary(merged, "latent_factor_period_mean", "inflation_rate_pct", max_lag=max_lag)
    inflation_to_factor = granger_summary(merged, "inflation_rate_pct", "latent_factor_period_mean", max_lag=max_lag)

    stats_payload = {
        "sample_start": str(merged.index.min().date()),
        "sample_end": str(merged.index.max().date()),
        "n_periods": int(len(merged)),
        "inflation_source": inflation_meta,
        "correlations": corr_stats,
        "adf_factor_period_mean": adf_factor,
        "adf_inflation_rate": adf_inflation,
        "granger_factor_to_inflation": factor_to_inflation,
        "granger_inflation_to_factor": inflation_to_factor,
    }
    write_json(stats_payload, ANALYSIS_DIR / "factor_inflation_stats.json")
    log(f"[output] wrote {(ANALYSIS_DIR / 'factor_inflation_stats.json').relative_to(ROOT_DIR).as_posix()}")

    write_summary(
        ANALYSIS_DIR / "factor_inflation_summary.txt",
        merged,
        inflation_meta,
        corr_stats,
        cross_corr,
        adf_factor,
        adf_inflation,
        factor_to_inflation,
        inflation_to_factor,
    )

    plot_timeseries(merged)
    plot_scatter(merged, corr_stats)
    plot_cross_correlation(cross_corr)

    log("[done] inflation analysis finished")


if __name__ == "__main__":
    main()
