from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trend import GuerreroTrendEstimator, forecast_trend, train_val_test_split


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "output"
ANALYSIS_DIR = OUTPUT_DIR / "trend_backtest_2008"
FACTOR_PATH = OUTPUT_DIR / "factor_latente_final.csv"

TRAIN_END = pd.Timestamp("2008-01-01")
BACKTEST_END = pd.Timestamp("2009-01-01")

D_ORDER = 2
TRAIN_FRAC = 0.75
VAL_FRAC = 0.25
MIN_TRAIN = 252
MIN_VAL = 63
SCAN_GRID = 120
REFINE_ITER = 15

CONFIDENCE_LEVELS = {
    "90.0%": 1.6448536269514722,
    "95.0%": 1.959963984540054,
    "97.5%": 2.241402727604947,
    "99.0%": 2.5758293035489004,
    "99.9%": 3.2905267314919255,
}


def log(message: str) -> None:
    print(message, flush=True)


def ensure_dirs() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    log(f"[plot] saved {path.relative_to(ROOT_DIR).as_posix()}")


def write_json(payload: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    log(f"[output] wrote {path.relative_to(ROOT_DIR).as_posix()}")


def load_factor_series(path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"Missing latent factor file: {path}")

    frame = pd.read_csv(path, parse_dates=["Date"])
    frame = frame.sort_values("Date").set_index("Date")
    if frame.empty:
        raise ValueError("Latent factor file is empty")

    series = pd.to_numeric(frame.iloc[:, 0], errors="coerce").dropna()
    if series.empty:
        raise ValueError("Latent factor series has no valid numeric observations")

    series.name = "latent_factor"
    return series


def split_train_and_backtest(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    train = series.loc[series.index < TRAIN_END].copy()
    backtest = series.loc[(series.index >= TRAIN_END) & (series.index < BACKTEST_END)].copy()

    if len(train) < MIN_TRAIN + MIN_VAL:
        raise ValueError(
            "Not enough pre-2008 observations to select the trend smoothness. "
            f"Need at least {MIN_TRAIN + MIN_VAL}, found {len(train)}."
        )
    if backtest.empty:
        raise ValueError("No observations were found in the requested backtest window.")

    return train, backtest


def select_smoothing(train: pd.Series) -> tuple[object, object, np.ndarray]:
    values = train.to_numpy(dtype=float)
    z_train, z_val, z_test, split = train_val_test_split(
        values,
        frac_train=TRAIN_FRAC,
        frac_val=VAL_FRAC,
        min_train=MIN_TRAIN,
        min_val=MIN_VAL,
    )

    estimator = GuerreroTrendEstimator(d=D_ORDER, n_train=split.n_train)
    scan = estimator.scan_local_minima(
        Z_train=z_train,
        Z_val=z_val,
        n_grid=SCAN_GRID,
        refine=True,
        refine_iter=REFINE_ITER,
    )
    best = scan.best()

    return scan, best, values


def fit_full_training(train: pd.Series, s_unit: float):
    values = train.to_numpy(dtype=float)
    estimator = GuerreroTrendEstimator(d=D_ORDER, n_train=len(values))
    fit = estimator.fit_train(values, s_unit=s_unit)
    return estimator, fit


def build_backtest_frame(train: pd.Series, backtest: pd.Series, fit) -> tuple[pd.DataFrame, float, float]:
    horizon = len(backtest)
    forecast_values = forecast_trend(fit.t_hat, d=D_ORDER, m_hat=fit.m_hat, h=horizon)

    residuals = train.to_numpy(dtype=float) - np.asarray(fit.t_hat, dtype=float)
    residual_std = float(np.std(residuals, ddof=1))
    innovations = np.diff(train.to_numpy(dtype=float))
    volatility = float(np.std(innovations, ddof=1))
    if not np.isfinite(volatility) or volatility <= 0.0:
        raise ValueError("Training volatility estimate is not positive.")

    steps = np.arange(1, horizon + 1, dtype=float)
    sqrt_h = np.sqrt(steps)

    frame = pd.DataFrame(
        {
            "Date": backtest.index,
            "actual": backtest.to_numpy(dtype=float),
            "forecast": forecast_values,
            "step": steps.astype(int),
            "sqrt_h": sqrt_h,
            "forecast_error": backtest.to_numpy(dtype=float) - forecast_values,
        }
    ).set_index("Date")

    for label, z_value in CONFIDENCE_LEVELS.items():
        suffix = label.replace("%", "").replace(".", "_")
        band = z_value * volatility * sqrt_h
        frame[f"lower_{suffix}"] = forecast_values - band
        frame[f"upper_{suffix}"] = forecast_values + band
        frame[f"outside_{suffix}"] = (frame["actual"] < frame[f"lower_{suffix}"]) | (
            frame["actual"] > frame[f"upper_{suffix}"]
        )

    return frame, volatility, residual_std


def plot_validation_scan(scan) -> None:
    minima_s = np.array([point.s_unit for point in scan.minima], dtype=float)
    minima_j = np.array([point.val_mse for point in scan.minima], dtype=float)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(scan.s_grid, scan.J_grid, linewidth=1.5, color="#1d4ed8", label="Validation MSE")
    if minima_s.size:
        ax.scatter(minima_s, minima_j, marker="*", s=85, color="black", label="Local minima")
    ax.set_title(f"Validation scan for smoothness selection (d={D_ORDER})")
    ax.set_xlabel("Smoothness index s")
    ax.set_ylabel("Validation MSE")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(frameon=False)
    save_figure(ANALYSIS_DIR / "validation_scan.png")


def plot_training_fit(estimator, train_values: np.ndarray, fit) -> None:
    poly_full = estimator.build_polynomial_full(
        t_hat_train=fit.t_hat,
        m_hat=fit.m_hat,
        n_total=len(train_values),
        n_train=len(train_values),
    )

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(train_values, linewidth=1.0, color="#64748b", label="Latent factor")
    ax.plot(fit.t_hat, linewidth=2.0, color="#b91c1c", label="Fitted trend")
    ax.plot(poly_full, linestyle="--", linewidth=1.4, color="#0f766e", label="Polynomial extension on train")
    ax.set_title("Trend fit on training sample up to 2008-01-01")
    ax.set_xlabel("Training observation index")
    ax.set_ylabel("Latent factor")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(frameon=False)
    save_figure(ANALYSIS_DIR / "training_fit.png")


def plot_backtest(frame: pd.DataFrame) -> None:
    ordered_levels = ["99.9%", "99.0%", "97.5%", "95.0%", "90.0%"]
    colors = {
        "99.9%": "#c7d2fe",
        "99.0%": "#a5b4fc",
        "97.5%": "#818cf8",
        "95.0%": "#6366f1",
        "90.0%": "#4338ca",
    }

    fig, ax = plt.subplots(figsize=(15, 6.5))
    for label in ordered_levels:
        suffix = label.replace("%", "").replace(".", "_")
        ax.fill_between(
            frame.index,
            frame[f"lower_{suffix}"],
            frame[f"upper_{suffix}"],
            color=colors[label],
            alpha=0.18,
            linewidth=0.0,
            label=f"{label} interval",
        )

    ax.plot(frame.index, frame["forecast"], color="#b91c1c", linewidth=2.0, label="Polynomial forecast")
    ax.plot(frame.index, frame["actual"], color="#111827", linewidth=1.2, label="Latent factor")
    ax.set_title("Latent factor backtest: 2008-01-01 to 2009-01-01")
    ax.set_xlabel("Date")
    ax.set_ylabel("Latent factor")
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.legend(loc="upper left", ncol=2, frameon=False)
    save_figure(ANALYSIS_DIR / "backtest_with_confidence_intervals.png")


def build_summary(
    train: pd.Series,
    backtest: pd.Series,
    best,
    fit,
    volatility: float,
    residual_std: float,
    frame: pd.DataFrame,
) -> tuple[dict, str]:
    interval_summary = {}
    summary_lines = [
        "Latent factor polynomial trend backtest",
        "=" * 38,
        "",
        f"Training start: {train.index.min().date()}",
        f"Training end: {(TRAIN_END - pd.Timedelta(days=1)).date()}",
        f"Backtest start: {backtest.index.min().date()}",
        f"Backtest end: {backtest.index.max().date()}",
        f"Training observations: {len(train)}",
        f"Backtest observations: {len(backtest)}",
        "",
        f"Difference order d: {D_ORDER}",
        f"Selected smoothness s: {best.s_unit:.6f}",
        f"Validation MSE at selected s: {best.val_mse:.6f}",
        f"Training change volatility for sqrt(h) bands: {volatility:.6f}",
        f"Training level residual std: {residual_std:.6f}",
        f"Estimated sigma^2 from full training fit: {fit.sigma2_hat:.6f}",
        "",
        "Out-of-band observations by confidence level:",
    ]

    for label in CONFIDENCE_LEVELS:
        suffix = label.replace("%", "").replace(".", "_")
        outside_count = int(frame[f"outside_{suffix}"].sum())
        outside_share = float(frame[f"outside_{suffix}"].mean())
        interval_summary[label] = {
            "outside_count": outside_count,
            "outside_share": outside_share,
        }
        summary_lines.append(
            f"{label}: {outside_count} dates ({outside_share:.2%} of the backtest window)"
        )

    largest_errors = (
        frame.assign(abs_error=lambda item: item["forecast_error"].abs())
        .sort_values("abs_error", ascending=False)
        .head(10)[["actual", "forecast", "forecast_error", "step"]]
    )

    summary_lines.extend(
        [
            "",
            "Largest absolute forecast errors:",
            largest_errors.to_string(float_format=lambda value: f"{value:.4f}"),
        ]
    )

    stats_payload = {
        "training_start": str(train.index.min().date()),
        "training_end_exclusive": str(TRAIN_END.date()),
        "backtest_start": str(backtest.index.min().date()),
        "backtest_end_exclusive": str(BACKTEST_END.date()),
        "n_train": int(len(train)),
        "n_backtest": int(len(backtest)),
        "difference_order": D_ORDER,
        "selected_s_unit": float(best.s_unit),
        "selected_validation_mse": float(best.val_mse),
        "training_change_volatility": float(volatility),
        "training_level_residual_std": float(residual_std),
        "training_sigma2_hat": float(fit.sigma2_hat),
        "interval_summary": interval_summary,
    }

    return stats_payload, "\n".join(summary_lines)


def main() -> None:
    ensure_dirs()
    log("[start] latent factor polynomial trend backtest")

    series = load_factor_series(FACTOR_PATH)
    log(f"[data] loaded latent factor with {len(series)} daily observations")

    train, backtest = split_train_and_backtest(series)
    log(f"[data] training window: {train.index.min().date()} to {train.index.max().date()} ({len(train)} obs)")
    log(f"[data] backtest window: {backtest.index.min().date()} to {backtest.index.max().date()} ({len(backtest)} obs)")

    scan, best, train_values = select_smoothing(train)
    minima_frame = pd.DataFrame(
        [{"s_unit": point.s_unit, "val_mse": point.val_mse} for point in scan.minima]
    )
    minima_frame.to_csv(ANALYSIS_DIR / "validation_scan_minima.csv", index=False)
    log(f"[output] wrote {(ANALYSIS_DIR / 'validation_scan_minima.csv').relative_to(ROOT_DIR).as_posix()}")

    train_estimator, fit = fit_full_training(train, s_unit=best.s_unit)
    frame, volatility, residual_std = build_backtest_frame(train, backtest, fit)
    frame.to_csv(ANALYSIS_DIR / "latent_factor_backtest_2008.csv", index_label="Date")
    log(f"[output] wrote {(ANALYSIS_DIR / 'latent_factor_backtest_2008.csv').relative_to(ROOT_DIR).as_posix()}")

    plot_validation_scan(scan)
    plot_training_fit(train_estimator, train_values, fit)
    plot_backtest(frame)

    stats_payload, summary_text = build_summary(train, backtest, best, fit, volatility, residual_std, frame)
    write_json(stats_payload, ANALYSIS_DIR / "latent_factor_backtest_2008_stats.json")

    summary_path = ANALYSIS_DIR / "latent_factor_backtest_2008_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    log(f"[output] wrote {summary_path.relative_to(ROOT_DIR).as_posix()}")

    log("[done] polynomial backtest finished")


if __name__ == "__main__":
    main()
