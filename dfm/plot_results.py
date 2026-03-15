from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator
from plot_style import PALETTE_2, PALETTE_3, PALETTE_5, PALETTE_7

plt.rcParams["font.family"] = "EB Garamond"
plt.rcParams["font.serif"] = ["EB Garamond", "serif"]


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "output"
PLOTS_DIR = OUTPUT_DIR / "plots"
FIGURE_SIZE = (10, 4)


def log(message: str) -> None:
    print(message, flush=True)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    frame = pd.read_csv(path, parse_dates=["Date"])
    frame = frame.sort_values("Date").set_index("Date")
    return frame


def load_metadata(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_current_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    log(f"[plot] saved {path.relative_to(ROOT_DIR).as_posix()}")


def apply_axis_style(ax, *, time_axis: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=6,
        width=1,
        colors="black",
        labelbottom=True,
        labelleft=True,
    )
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    if time_axis:
        locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))


def set_title_and_legend(ax, title: str, *, ncol: int = 1) -> None:
    ax.set_title(title, pad=1)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=ncol,
            frameon=False,
        )


def plot_latent_factor(factor: pd.DataFrame) -> None:
    series = factor.iloc[:, 0].rename("latent_factor")
    rolling = series.rolling(60, min_periods=20).mean()

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    fig.patch.set_facecolor("#E0E1E5")
    ax.set_facecolor("#E0E1E5")

    ax.plot(series.index, series.values, color=PALETTE_3, linewidth=1.0, alpha=0.75, label="Factor latente")
    ax.plot(rolling.index, rolling.values, color=PALETTE_7, linewidth=2.0, label="Media móvil 60 días")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    set_title_and_legend(ax, "Factor latente final", ncol=2)
    ax.set_ylabel("Nivel")
    apply_axis_style(ax)
    ax.grid(alpha=0.25)
    save_current_figure(PLOTS_DIR / "latent_factor.png")


def plot_original_series(raw_panel: pd.DataFrame) -> None:
    indexed = raw_panel.divide(raw_panel.iloc[0]).multiply(100.0)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    fig.patch.set_facecolor("#E0E1E5")
    ax.set_facecolor("#E0E1E5")

    for column in indexed.columns:
        ax.plot(indexed.index, indexed[column], linewidth=1.3, label=column)

    set_title_and_legend(ax, "Series originales indexadas a 100", ncol=2)
    ax.set_ylabel("Índice base 100")
    apply_axis_style(ax)
    ax.grid(alpha=0.25)

    save_current_figure(PLOTS_DIR / "series_originales_indexadas.png")


def plot_transformed_series(transformed_panel: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        transformed_panel.shape[1],
        1,
        figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1] * transformed_panel.shape[1]),
        sharex=True,
    )

    fig.patch.set_facecolor("#E0E1E5")

    if transformed_panel.shape[1] == 1:
        axes = [axes]

    for ax, column in zip(axes, transformed_panel.columns):
        ax.set_facecolor("#E0E1E5")
        ax.plot(
            transformed_panel.index,
            transformed_panel[column],
            linewidth=1.0,
            color=PALETTE_2,
        )
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        ax.set_title(f"{column} transformada y estandarizada", pad=20)
        apply_axis_style(ax)
        ax.grid(alpha=0.25)

    save_current_figure(PLOTS_DIR / "series_transformadas.png")


def plot_factor_vs_inputs(factor: pd.DataFrame, transformed_panel: pd.DataFrame) -> None:
    merged = transformed_panel.copy()
    merged["latent_factor"] = factor.iloc[:, 0]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    fig.patch.set_facecolor("#E0E1E5")
    ax.set_facecolor("#E0E1E5")

    ax.plot(
        merged.index,
        merged["latent_factor"],
        color=PALETTE_7,
        linewidth=2.2,
        label="Factor latente",
    )

    for column in transformed_panel.columns:
        ax.plot(
            merged.index,
            merged[column],
            linewidth=0.9,
            alpha=0.65,
            label=column,
        )

    set_title_and_legend(
        ax,
        "Factor latente vs series transformadas",
        ncol=len(transformed_panel.columns) + 1,
        
    )
    apply_axis_style(ax)
    ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(alpha=0.25)
    ax.set_ylabel("Nivel estandarizado")
    ax.set_xlabel("Fecha")

    save_current_figure(PLOTS_DIR / "factor_vs_series_transformadas.png")


def plot_diagnostics(metadata: Dict) -> None:
    details = metadata.get("diagnostics", {}).get("detail", {})
    if not details:
        log("[plot] no diagnostics detail found; skipping diagnostics figure")
        return

    diagnostics_frame = pd.DataFrame(details).T[
        ["ljung_box_pvalue", "jarque_bera_pvalue", "arch_pvalue"]
    ]

    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZE, sharey=True)
    fig.patch.set_facecolor("#E0E1E5")

    titles = {
        "ljung_box_pvalue": "Valor p de Ljung-Box",
        "jarque_bera_pvalue": "Valor p de Jarque-Bera",
        "arch_pvalue": "Valor p de ARCH",
    }
    colors = {
        "ljung_box_pvalue": PALETTE_3,
        "jarque_bera_pvalue": PALETTE_7,
        "arch_pvalue": PALETTE_5,
    }

    for ax, column in zip(axes, diagnostics_frame.columns):
        ax.set_facecolor("#E0E1E5")
        ax.barh(
            diagnostics_frame.index,
            diagnostics_frame[column],
            color=colors[column],
            alpha=0.85,
        )
        ax.axvline(0.05, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.set_title(titles[column], pad=20)
        apply_axis_style(ax, time_axis=False)
        ax.grid(axis="x", alpha=0.25)

    fig.suptitle("Diagnósticos del mejor modelo", fontsize=15, y=1.02)
    save_current_figure(PLOTS_DIR / "diagnosticos_mejor_modelo.png")


def plot_dashboard(
    raw_panel: pd.DataFrame,
    transformed_panel: pd.DataFrame,
    factor: pd.DataFrame,
    metadata: Dict,
) -> None:
    factor_series = factor.iloc[:, 0]
    rolling = factor_series.rolling(60, min_periods=20).mean()
    indexed = raw_panel.divide(raw_panel.iloc[0]).multiply(100.0)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.patch.set_facecolor("#E0E1E5")

    for ax in axes.flatten():
        ax.set_facecolor("#E0E1E5")
        apply_axis_style(ax)

    axes[0, 0].plot(
        factor_series.index,
        factor_series.values,
        color=PALETTE_3,
        linewidth=1.0,
        label="Factor latente",
    )
    axes[0, 0].plot(
        rolling.index,
        rolling.values,
        color=PALETTE_7,
        linewidth=2.0,
        label="Media móvil 60 días",
    )
    axes[0, 0].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    set_title_and_legend(axes[0, 0], "Factor latente", ncol=2)
    axes[0, 0].grid(alpha=0.25)

    for column in indexed.columns:
        axes[0, 1].plot(indexed.index, indexed[column], linewidth=1.1, label=column)
    set_title_and_legend(axes[0, 1], "Series originales indexadas", ncol=2)
    axes[0, 1].grid(alpha=0.25)

    for column in transformed_panel.columns:
        axes[1, 0].plot(
            transformed_panel.index,
            transformed_panel[column],
            linewidth=0.9,
            alpha=0.8,
            label=column,
        )
    set_title_and_legend(axes[1, 0], "Series transformadas", ncol=2)
    axes[1, 0].grid(alpha=0.25)

    best_model = metadata.get("best_model", {})
    text_lines = [
        "Mejor modelo",
        f"Número de factores: {best_model.get('k_factors')}",
        f"Orden del factor: {best_model.get('factor_order')}",
        f"Orden del error: {best_model.get('error_order')}",
        f"Covarianza: {best_model.get('error_cov_type')}",
        f"AIC: {best_model.get('aic')}",
        f"Convergencia: {best_model.get('converged')}",
        f"Diagnósticos aprobados: {best_model.get('diagnostics_passed_count')}/4",
    ]

    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.02,
        0.98,
        "\n".join(text_lines),
        va="top",
        ha="left",
        fontsize=12,
    )

    fig.suptitle("Resumen visual del DFM", fontsize=16)
    save_current_figure(PLOTS_DIR / "dashboard_dfm.png")


def main() -> None:
    log("[start] loading DFM outputs for plotting")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    factor = load_csv(OUTPUT_DIR / "factor_latente_final.csv")
    raw_panel = load_csv(OUTPUT_DIR / "panel_original_niveles.csv")
    transformed_panel = load_csv(OUTPUT_DIR / "panel_transformado_usado.csv")
    metadata = load_metadata(OUTPUT_DIR / "dfm_run_metadata.json")

    plot_latent_factor(factor)
    plot_original_series(raw_panel)
    plot_transformed_series(transformed_panel)
    plot_factor_vs_inputs(factor, transformed_panel)
    plot_diagnostics(metadata)
    plot_dashboard(raw_panel, transformed_panel, factor, metadata)

    log("[done] all plots generated")


if __name__ == "__main__":
    main()
