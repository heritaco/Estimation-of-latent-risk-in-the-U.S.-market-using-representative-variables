from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import FIGURE_SIZE, PALETTE_1, PALETTE_2, PALETTE_3, PALETTE_4, PALETTE_5, PALETTE_6, PALETTE_7, style_axis, style_figure, style_grid, style_legend, style_text_box
from trend import GuerreroTrendEstimator, forecast_trend


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "output"
ANALYSIS_DIR = OUTPUT_DIR / "trend_backtest_2008"
FACTOR_PATH = OUTPUT_DIR / "factor_latente_final.csv"

TRAIN_END = pd.Timestamp("2008-01-01")
VALIDATION_END = pd.Timestamp("2008-06-01")
TEST_END = pd.Timestamp("2009-01-01")

D_ORDER = 2
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


def split_windows(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    train = series.loc[series.index < TRAIN_END].copy()
    validation = series.loc[(series.index >= TRAIN_END) & (series.index < VALIDATION_END)].copy()
    test = series.loc[(series.index >= VALIDATION_END) & (series.index < TEST_END)].copy()

    if len(train) < MIN_TRAIN:
        raise ValueError(
            "Not enough pre-2008 observations for training. "
            f"Need at least {MIN_TRAIN}, found {len(train)}."
        )
    if len(validation) < MIN_VAL:
        raise ValueError(
            "Not enough observations in the validation window. "
            f"Need at least {MIN_VAL}, found {len(validation)}."
        )
    if test.empty:
        raise ValueError("No observations were found in the requested test window.")

    return train, validation, test


def select_smoothing(train: pd.Series, validation: pd.Series) -> tuple[object, dict[str, float], np.ndarray]:
    train_values = train.to_numpy(dtype=float)
    validation_values = validation.to_numpy(dtype=float)

    estimator = GuerreroTrendEstimator(d=D_ORDER, n_train=len(train_values))
    scan = estimator.scan_local_minima(
        Z_train=train_values,
        Z_val=validation_values,
        n_grid=SCAN_GRID,
        refine=True,
        refine_iter=REFINE_ITER,
    )
    if not scan.minima:
        raise ValueError("No local minima were found in the smoothness scan.")
    first_local = scan.minima[0]
    global_idx = int(np.argmin(scan.J_grid))
    selection = {
        "selected_s_unit": float(scan.s_grid[global_idx]),
        "selected_val_mse": float(scan.J_grid[global_idx]),
        "first_local_s_unit": float(first_local.s_unit),
        "first_local_val_mse": float(first_local.val_mse),
        "global_best_s_unit": float(scan.s_grid[global_idx]),
        "global_best_val_mse": float(scan.J_grid[global_idx]),
    }

    return scan, selection, train_values


def fit_window(history: pd.Series, s_unit: float):
    history_values = history.to_numpy(dtype=float)
    estimator = GuerreroTrendEstimator(d=D_ORDER, n_train=len(history_values))
    fit = estimator.fit_train(history_values, s_unit=s_unit)
    return estimator, fit


def build_forecast_frame(
    target: pd.Series,
    forecast_values: np.ndarray,
    volatility: float,
    residual_std: float,
    step_offset: int,
    include_intervals: bool,
) -> pd.DataFrame:
    horizon = len(target)
    steps = np.arange(step_offset + 1, step_offset + horizon + 1, dtype=float)
    sqrt_h = np.sqrt(steps)

    frame = pd.DataFrame(
        {
            "Date": target.index,
            "actual": target.to_numpy(dtype=float),
            "forecast": forecast_values,
            "step": steps.astype(int),
            "sqrt_h": sqrt_h,
            "forecast_error": target.to_numpy(dtype=float) - forecast_values,
        }
    ).set_index("Date")

    if include_intervals:
        for label, z_value in CONFIDENCE_LEVELS.items():
            suffix = label.replace("%", "").replace(".", "_")
            band = z_value * volatility * sqrt_h
            frame[f"lower_{suffix}"] = forecast_values - band
            frame[f"upper_{suffix}"] = forecast_values + band
            frame[f"outside_{suffix}"] = (frame["actual"] < frame[f"lower_{suffix}"]) | (
                frame["actual"] > frame[f"upper_{suffix}"]
            )

    return frame


def compute_training_uncertainty(history: pd.Series, fit) -> tuple[float, float]:
    history_values = history.to_numpy(dtype=float)
    residuals = history_values - np.asarray(fit.t_hat, dtype=float)
    residual_std = float(np.std(residuals, ddof=1))
    innovations = np.diff(history_values)
    volatility = float(np.std(innovations, ddof=1))
    if not np.isfinite(volatility) or volatility <= 0.0:
        raise ValueError("Training volatility estimate is not positive.")
    return volatility, residual_std


def plot_validation_scan(scan, selection: dict[str, float]) -> None:
    minima_s = np.array([point.s_unit for point in scan.minima], dtype=float)
    minima_j = np.array([point.val_mse for point in scan.minima], dtype=float)
    first_local_s = selection["first_local_s_unit"]
    first_local_j = selection["first_local_val_mse"]
    selected_s = selection["selected_s_unit"]
    selected_j = selection["selected_val_mse"]

    fig, ax = plt.subplots(figsize=(10, 8/3))
    style_figure(fig)
    style_axis(ax)
    ax.plot(scan.s_grid, scan.J_grid, linewidth=1.5, color=PALETTE_1, label="Error cuadrático medio de validación")
    if minima_s.size:
        ax.scatter(minima_s, minima_j, marker="*", s=85, color="black", label="Mínimos locales")
    ax.scatter(
        [selected_s],
        [selected_j],
        marker="o",
        s=55,
        color=PALETTE_7,
        label="Mejor global",
        zorder=3,
    )
    ax.set_title(f"Barrido de validación para elegir suavidad (d={D_ORDER})")
    ax.set_xlabel("Índice de suavidad s")
    ax.set_ylabel("MSE de validación")
    style_grid(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False, ncol=3)
    # style_text_box(
    #     ax,
    #     (
    #         f"Primer minimo local: s = {first_local_s:.6f}, MSE = {first_local_j:.6f}\n"
    #         f"Mejor global: s = {selected_s:.6f}, MSE = {selected_j:.6f}\n"
    #         f"Criterio usado: mejor global"
    #     ),
    #     x=0.98,
    #     y=0.03,
    #     va="bottom",
    #     ha="right",
    # )
    save_figure(ANALYSIS_DIR / "validation_scan.png")


def plot_training_fit(estimator, train_values: np.ndarray, fit, selected_s: float) -> None:
    poly_full = estimator.build_polynomial_full(
        t_hat_train=fit.t_hat,
        m_hat=fit.m_hat,
        n_total=len(train_values),
        n_train=len(train_values),
    )

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    style_figure(fig)
    style_axis(ax)
    ax.plot(train_values, linewidth=1.0, color=PALETTE_4, label="Factor latente")
    ax.plot(fit.t_hat, linewidth=2.0, color=PALETTE_7, label="Tendencia ajustada")
    ax.plot(poly_full, linestyle="--", linewidth=1.4, color=PALETTE_2, label="Polinomio sobre entrenamiento")
    ax.set_title("Ajuste de tendencia en la muestra de entrenamiento")
    ax.set_xlabel("Índice de observación en entrenamiento")
    ax.set_ylabel("Factor latente")
    style_grid(ax)
    style_legend(ax)
    style_text_box(
        ax,
        (
            f"Período: {train_values.size} observaciones hasta 2007-12-31\n"
            f"d = {D_ORDER}\n"
            f"s elegido = {selected_s:.6f}\n"
            f"sigma^2 estimada = {fit.sigma2_hat:.6f}"
        ),
    )
    save_figure(ANALYSIS_DIR / "training_fit.png")


def plot_validation_forecast(frame: pd.DataFrame, selected_s: float, validation_rmse: float) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    style_figure(fig)
    style_axis(ax, time_axis=True)
    ax.plot(frame.index, frame["forecast"], color=PALETTE_7, linewidth=2.0, label="Pronóstico polinomial")
    ax.plot(frame.index, frame["actual"], color=PALETTE_4, linewidth=1.2, label="Factor latente")
    ax.set_title("Pronóstico en validación: 2008-01-01 a 2008-06-01")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Factor latente")
    style_grid(ax)
    style_legend(ax)
    style_text_box(
        ax,
        (
            f"Origen del pronóstico: entrenamiento\n"
            f"d = {D_ORDER}\n"
            f"s elegido = {selected_s:.6f}\n"
            f"Raíz del ECM de validación = {validation_rmse:.6f}"
        ),
    )
    save_figure(ANALYSIS_DIR / "validation_forecast.png")


def plot_test_forecast(
    frame: pd.DataFrame,
    selected_s: float,
    training_volatility: float,
    test_rmse: float,
) -> None:
    ordered_levels = ["99.9%", "99.0%", "97.5%", "95.0%", "90.0%"]
    colors = {
        "99.9%": PALETTE_1,
        "99.0%": PALETTE_2,
        "97.5%": PALETTE_3,
        "95.0%": PALETTE_4,
        "90.0%": PALETTE_5,
    }

    fig, ax = plt.subplots(figsize=(10, 4))
    style_figure(fig)
    style_axis(ax, time_axis=True)
    for label in ordered_levels:
        suffix = label.replace("%", "").replace(".", "_")
        ax.fill_between(
            frame.index,
            frame[f"lower_{suffix}"],
            frame[f"upper_{suffix}"],
            color=colors[label],
            alpha=0.18,
            linewidth=0.0,
            label=f"Intervalo {label}",
        )

    ax.plot(frame.index, frame["forecast"], color=PALETTE_7, linewidth=2.0, label="Pronóstico polinomial")
    ax.plot(frame.index, frame["actual"], color=PALETTE_4, linewidth=1.2, label="Factor latente")
    ax.set_title("Pronóstico en prueba con intervalos de confianza")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Factor latente")
    style_grid(ax)
    style_legend(ax, ncol=2)
    style_text_box(
        ax,
        (
            f"El polinomio continúa desde validación\n"
            f"d = {D_ORDER}\n"
            f"s elegido = {selected_s:.6f}\n"
            f"Volatilidad usada en sqrt(h) = {training_volatility:.6f}\n"
            f"Raíz del ECM de prueba = {test_rmse:.6f}"
        ),
    )
    save_figure(ANALYSIS_DIR / "test_with_confidence_intervals.png")


def plot_full_timeline(
    train: pd.Series,
    train_fit,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    selected_s: float,
    validation_rmse: float,
    test_rmse: float,
) -> None:
    ordered_levels = ["99.9%", "99.0%", "97.5%", "95.0%", "90.0%"]
    colors = {
        "99.9%": "#afd3ff",
        "99.0%": "#8abeff",
        "97.5%": "#53a0ff",
        "95.0%": "#3590ff",
        "90.0%": "#2b8aff",
    }

    fig, ax = plt.subplots(figsize=(10, 4))
    style_figure(fig)
    style_axis(ax, time_axis=True)
    validation_mse = validation_rmse ** 2

    ax.plot(train.index, train.to_numpy(dtype=float), color=PALETTE_1, linewidth=1.8, label="Entrenamiento", alpha = 0.5)
    ax.plot(
        train.index,
        np.asarray(train_fit.t_hat, dtype=float),
        color=PALETTE_7,
        linewidth=1.5,
        label=f"Tendencia ajustada (s={selected_s:.4f})",
    )

    ax.plot(
        validation_frame.index,
        validation_frame["actual"],
        color=PALETTE_2,
        linewidth=1.2,
        label="Validación",
    )
    ax.plot(
        validation_frame.index,
        validation_frame["forecast"],
        color=PALETTE_7,
        linewidth=1.8,
        linestyle="--",
        label=f"Pronóstico validación (MSE={validation_mse:.4f})",
    )

    for label in ordered_levels:
        suffix = label.replace("%", "").replace(".", "_")
        ax.fill_between(
            test_frame.index,
            test_frame[f"lower_{suffix}"],
            test_frame[f"upper_{suffix}"],
            color=colors[label],
            alpha=0.18,
            linewidth=0.0,
        )

    ax.plot(test_frame.index, test_frame["actual"], color=PALETTE_3, linewidth=1.3, label="Prueba")
    ax.plot(test_frame.index, test_frame["forecast"], color=PALETTE_7, linewidth=2.0, label="Pronóstico prueba")

    ax.axvline(TRAIN_END, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axvline(VALIDATION_END, color="black", linestyle=":", linewidth=1.0, alpha=0.8)

    ax.set_title("Serie completa: entrenamiento, validación y prueba")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Factor latente")
    style_grid(ax)

    handles, labels = ax.get_legend_handles_labels()
    legend_map = dict(zip(labels, handles))
    ordered_labels = [
        "Entrenamiento",
        f"Tendencia ajustada (s={selected_s:.4f})",
        "Validación",
        f"Pronóstico validación (MSE={validation_mse:.4f})",
        "Prueba",
        "Pronóstico prueba",
    ]
    ordered_handles = [legend_map[label] for label in ordered_labels]

    ax.legend(
        ordered_handles,
        ordered_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
        ncol=3,
    )
    # style_text_box(
    #     ax,
    #     (
    #         f"Indice de suavidad elegido: s = {selected_s:.6f}\n"
    #         f"d = {D_ORDER}\n"
    #         f"Raiz del ECM de validacion = {validation_rmse:.6f}\n"
    #         f"Raiz del ECM de prueba = {test_rmse:.6f}"
    #     ),
    #     x=0.02,
    #     y=0.03,
    #     va="bottom",
    # )
    save_figure(ANALYSIS_DIR / "full_timeline_train_validation_test.png")


def build_summary(
    train: pd.Series,
    validation: pd.Series,
    test: pd.Series,
    selection: dict[str, float],
    train_fit,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    training_volatility: float,
    training_residual_std: float,
) -> tuple[dict, str]:
    interval_summary = {}
    validation_rmse = float(np.sqrt(np.mean(np.square(validation_frame["forecast_error"]))))
    test_rmse = float(np.sqrt(np.mean(np.square(test_frame["forecast_error"]))))
    summary_lines = [
        "Latent factor train / validation / test backtest",
        "=" * 48,
        "",
        f"Training start: {train.index.min().date()}",
        f"Training end: {(TRAIN_END - pd.Timedelta(days=1)).date()}",
        f"Validation start: {validation.index.min().date()}",
        f"Validation end: {validation.index.max().date()}",
        f"Test start: {test.index.min().date()}",
        f"Test end: {test.index.max().date()}",
        f"Training observations: {len(train)}",
        f"Validation observations: {len(validation)}",
        f"Test observations: {len(test)}",
        "",
        f"Difference order d: {D_ORDER}",
        f"Selected smoothness s (global best): {selection['selected_s_unit']:.6f}",
        f"Validation MSE at selected s: {selection['selected_val_mse']:.6f}",
        f"First local minimum s: {selection['first_local_s_unit']:.6f}",
        f"Validation MSE at first local minimum: {selection['first_local_val_mse']:.6f}",
        f"Validation RMSE using the training polynomial forecast: {validation_rmse:.6f}",
        f"Test RMSE using the same training polynomial forecast: {test_rmse:.6f}",
        f"Train sigma^2: {train_fit.sigma2_hat:.6f}",
        f"Training change volatility for sqrt(h) bands: {training_volatility:.6f}",
        f"Training level residual std: {training_residual_std:.6f}",
        "Confidence intervals are drawn only on the test window and restart at h=1 on the first test date.",
        "",
        "Test out-of-band observations by confidence level:",
    ]

    for label in CONFIDENCE_LEVELS:
        suffix = label.replace("%", "").replace(".", "_")
        outside_count = int(test_frame[f"outside_{suffix}"].sum())
        outside_share = float(test_frame[f"outside_{suffix}"].mean())
        interval_summary[label] = {
            "outside_count": outside_count,
            "outside_share": outside_share,
        }
        summary_lines.append(
            f"{label}: {outside_count} dates ({outside_share:.2%} of the test window)"
        )

    largest_validation_errors = (
        validation_frame.assign(abs_error=lambda item: item["forecast_error"].abs())
        .sort_values("abs_error", ascending=False)
        .head(5)[["actual", "forecast", "forecast_error", "step"]]
    )
    largest_test_errors = (
        test_frame.assign(abs_error=lambda item: item["forecast_error"].abs())
        .sort_values("abs_error", ascending=False)
        .head(10)[["actual", "forecast", "forecast_error", "step"]]
    )

    summary_lines.extend(
        [
            "",
            "Largest validation forecast errors:",
            largest_validation_errors.to_string(float_format=lambda value: f"{value:.4f}"),
            "",
            "Largest test forecast errors:",
            largest_test_errors.to_string(float_format=lambda value: f"{value:.4f}"),
        ]
    )

    stats_payload = {
        "training_start": str(train.index.min().date()),
        "training_end_exclusive": str(TRAIN_END.date()),
        "validation_start": str(validation.index.min().date()),
        "validation_end_exclusive": str(VALIDATION_END.date()),
        "test_start": str(test.index.min().date()),
        "test_end_exclusive": str(TEST_END.date()),
        "n_train": int(len(train)),
        "n_validation": int(len(validation)),
        "n_test": int(len(test)),
        "difference_order": D_ORDER,
        "selection_rule": "global_best_with_first_local_reported",
        "selected_s_unit": float(selection["selected_s_unit"]),
        "selected_validation_mse": float(selection["selected_val_mse"]),
        "first_local_s_unit": float(selection["first_local_s_unit"]),
        "first_local_validation_mse": float(selection["first_local_val_mse"]),
        "validation_rmse": validation_rmse,
        "test_rmse": test_rmse,
        "train_sigma2_hat": float(train_fit.sigma2_hat),
        "training_change_volatility": float(training_volatility),
        "training_level_residual_std": float(training_residual_std),
        "test_interval_summary": interval_summary,
    }

    return stats_payload, "\n".join(summary_lines)


def main() -> None:
    ensure_dirs()
    log("[start] latent factor polynomial trend backtest")

    series = load_factor_series(FACTOR_PATH)
    log(f"[data] loaded latent factor with {len(series)} daily observations")

    train, validation, test = split_windows(series)
    log(f"[data] training window: {train.index.min().date()} to {train.index.max().date()} ({len(train)} obs)")
    log(f"[data] validation window: {validation.index.min().date()} to {validation.index.max().date()} ({len(validation)} obs)")
    log(f"[data] test window: {test.index.min().date()} to {test.index.max().date()} ({len(test)} obs)")

    scan, selection, train_values = select_smoothing(train, validation)
    selected_s_unit = float(selection["selected_s_unit"])
    minima_frame = pd.DataFrame(
        [
            {
                "s_unit": point.s_unit,
                "val_mse": point.val_mse,
                "is_first_local": bool(np.isclose(point.s_unit, selection["first_local_s_unit"])),
                "is_selected_global": bool(np.isclose(point.s_unit, selection["selected_s_unit"])),
            }
            for point in scan.minima
        ]
    )
    minima_frame.to_csv(ANALYSIS_DIR / "validation_scan_minima.csv", index=False)
    log(f"[output] wrote {(ANALYSIS_DIR / 'validation_scan_minima.csv').relative_to(ROOT_DIR).as_posix()}")

    train_estimator, train_fit = fit_window(train, s_unit=selected_s_unit)
    training_volatility, training_residual_std = compute_training_uncertainty(train, train_fit)

    total_horizon = len(validation) + len(test)
    full_forecast = forecast_trend(train_fit.t_hat, d=D_ORDER, m_hat=train_fit.m_hat, h=total_horizon)

    validation_forecast = full_forecast[: len(validation)]
    test_forecast = full_forecast[len(validation) :]

    validation_frame = build_forecast_frame(
        validation,
        validation_forecast,
        training_volatility,
        training_residual_std,
        step_offset=0,
        include_intervals=False,
    )
    validation_frame.to_csv(ANALYSIS_DIR / "latent_factor_validation_2008.csv", index_label="Date")
    log(f"[output] wrote {(ANALYSIS_DIR / 'latent_factor_validation_2008.csv').relative_to(ROOT_DIR).as_posix()}")

    test_frame = build_forecast_frame(
        test,
        test_forecast,
        training_volatility,
        training_residual_std,
        step_offset=0,
        include_intervals=True,
    )
    test_frame.to_csv(ANALYSIS_DIR / "latent_factor_test_2008.csv", index_label="Date")
    log(f"[output] wrote {(ANALYSIS_DIR / 'latent_factor_test_2008.csv').relative_to(ROOT_DIR).as_posix()}")

    plot_validation_scan(scan, selection)
    validation_rmse = float(np.sqrt(np.mean(np.square(validation_frame["forecast_error"]))))
    test_rmse = float(np.sqrt(np.mean(np.square(test_frame["forecast_error"]))))

    plot_training_fit(train_estimator, train_values, train_fit, selected_s_unit)
    plot_validation_forecast(validation_frame, selected_s_unit, validation_rmse)
    plot_test_forecast(test_frame, selected_s_unit, training_volatility, test_rmse)
    plot_full_timeline(
        train,
        train_fit,
        validation_frame,
        test_frame,
        selected_s_unit,
        validation_rmse,
        test_rmse,
    )

    stats_payload, summary_text = build_summary(
        train,
        validation,
        test,
        selection,
        train_fit,
        validation_frame,
        test_frame,
        training_volatility,
        training_residual_std,
    )
    write_json(stats_payload, ANALYSIS_DIR / "latent_factor_backtest_2008_stats.json")

    summary_path = ANALYSIS_DIR / "latent_factor_backtest_2008_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    log(f"[output] wrote {summary_path.relative_to(ROOT_DIR).as_posix()}")

    log("[done] polynomial backtest finished")


if __name__ == "__main__":
    main()
