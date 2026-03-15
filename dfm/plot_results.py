from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "output"
PLOTS_DIR = OUTPUT_DIR / "plots"


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


def plot_latent_factor(factor: pd.DataFrame) -> None:
    series = factor.iloc[:, 0].rename("latent_factor")
    rolling = series.rolling(60, min_periods=20).mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#E0E1E5")
    ax.set_facecolor("#E0E1E5")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
    axis="both",
    which="major",
    direction="out",
    length=6,
    width=1,
    colors="black"
)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.plot(series.index, series.values, color="#1f4e79", linewidth=1.0, alpha=0.75, label="Factor latente")
    ax.plot(rolling.index, rolling.values, color="#c0392b", linewidth=2.0, label="Media movil 60 dias")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_title("Factor latente final")
    ax.set_ylabel("Nivel")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.25)
    save_current_figure(PLOTS_DIR / "latent_factor.png")


def plot_original_series(raw_panel: pd.DataFrame) -> None:
    indexed = raw_panel.divide(raw_panel.iloc[0]).multiply(100.0)

    fig, ax = plt.subplots(figsize=(14, 6))

    fig.patch.set_facecolor("#E0E1E5")
    ax.set_facecolor("#E0E1E5")

    for column in indexed.columns:
        ax.plot(indexed.index, indexed[column], linewidth=1.3, label=column)

    ax.set_title("Series originales indexadas a 100")
    ax.set_ylabel("Indice base 100")

    # quitar bordes superior y derecho
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # mejorar bordes izquierdo e inferior
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    # ticks estilo académico
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=6,
        width=1,
        colors="black"
    )

    # leyenda sin fondo
    ax.legend(loc="upper left", ncol=2, frameon=False)

    ax.grid(alpha=0.25)

    save_current_figure(PLOTS_DIR / "series_originales_indexadas.png")


def plot_transformed_series(transformed_panel: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        transformed_panel.shape[1],
        1,
        figsize=(14, 2.6 * transformed_panel.shape[1]),
        sharex=True
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
            color="#0b6e4f"
        )

        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)

        ax.set_title(f"{column} transformada y estandarizada")

        # quitar bordes superior y derecho
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # mejorar bordes izquierdo e inferior
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)

        # ticks estilo académico
        ax.tick_params(
            axis="both",
            which="major",
            direction="out",
            length=6,
            width=1,
            colors="black"
        )

        ax.grid(alpha=0.25)

    save_current_figure(PLOTS_DIR / "series_transformadas.png")


def plot_factor_vs_inputs(factor: pd.DataFrame, transformed_panel: pd.DataFrame) -> None:
    merged = transformed_panel.copy()
    merged["latent_factor"] = factor.iloc[:, 0]

    fig, ax = plt.subplots(figsize=(14, 6))

    fig.patch.set_facecolor("#E0E1E5")
    ax.set_facecolor("#E0E1E5")

    ax.plot(
        merged.index,
        merged["latent_factor"],
        color="#7f1d1d",
        linewidth=2.2,
        label="Factor latente"
    )

    for column in transformed_panel.columns:
        ax.plot(
            merged.index,
            merged[column],
            linewidth=0.9,
            alpha=0.65,
            label=column
        )

    ax.set_title("Factor latente vs series transformadas")

    # quitar bordes superior y derecho
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # mejorar bordes izquierdo e inferior
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    # ticks estilo académico
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=6,
        width=1,
        colors="black"
    )

    # leyenda sin fondo
    ax.legend(loc="upper left", ncol=3, frameon=False)

    ax.grid(alpha=0.25)

    save_current_figure(PLOTS_DIR / "factor_vs_series_transformadas.png")


def plot_diagnostics(metadata: Dict) -> None:
    details = metadata.get("diagnostics", {}).get("detail", {})
    if not details:
        log("[plot] no diagnostics detail found; skipping diagnostics figure")
        return

    diagnostics_frame = pd.DataFrame(details).T[
        ["ljung_box_pvalue", "jarque_bera_pvalue", "arch_pvalue"]
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    fig.patch.set_facecolor("#E0E1E5")

    titles = {
        "ljung_box_pvalue": "Ljung-Box p-value",
        "jarque_bera_pvalue": "Jarque-Bera p-value",
        "arch_pvalue": "ARCH p-value",
    }
    colors = {
        "ljung_box_pvalue": "#2563eb",
        "jarque_bera_pvalue": "#dc2626",
        "arch_pvalue": "#7c3aed",
    }

    for ax, column in zip(axes, diagnostics_frame.columns):
        ax.set_facecolor("#E0E1E5")

        ax.barh(
            diagnostics_frame.index,
            diagnostics_frame[column],
            color=colors[column],
            alpha=0.85
        )
        ax.axvline(0.05, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.set_title(titles[column])

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
            colors="black"
        )

        ax.grid(axis="x", alpha=0.25)

    fig.suptitle("Diagnosticos del mejor modelo")
    save_current_figure(PLOTS_DIR / "diagnosticos_mejor_modelo.png")


def plot_dashboard(raw_panel: pd.DataFrame, transformed_panel: pd.DataFrame, factor: pd.DataFrame, metadata: Dict) -> None:
    factor_series = factor.iloc[:, 0]
    rolling = factor_series.rolling(60, min_periods=20).mean()
    indexed = raw_panel.divide(raw_panel.iloc[0]).multiply(100.0)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    fig.patch.set_facecolor("#E0E1E5")

    # aplicar fondo y estilo a todos los subplots
    for ax in axes.flatten():
        ax.set_facecolor("#E0E1E5")

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
            colors="black"
        )

    axes[0, 0].plot(factor_series.index, factor_series.values, color="#1d4ed8", linewidth=1.0)
    axes[0, 0].plot(rolling.index, rolling.values, color="#b91c1c", linewidth=2.0)
    axes[0, 0].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axes[0, 0].set_title("Factor latente")
    axes[0, 0].grid(alpha=0.25)

    for column in indexed.columns:
        axes[0, 1].plot(indexed.index, indexed[column], linewidth=1.1, label=column)
    axes[0, 1].set_title("Series originales indexadas")
    axes[0, 1].legend(loc="upper left", ncol=2, frameon=False)
    axes[0, 1].grid(alpha=0.25)

    for column in transformed_panel.columns:
        axes[1, 0].plot(transformed_panel.index, transformed_panel[column], linewidth=0.9, alpha=0.8, label=column)
    axes[1, 0].set_title("Series transformadas")
    axes[1, 0].legend(loc="upper left", ncol=2, frameon=False)
    axes[1, 0].grid(alpha=0.25)

    best_model = metadata.get("best_model", {})
    text_lines = [
        "Mejor modelo",
        f"k_factors: {best_model.get('k_factors')}",
        f"factor_order: {best_model.get('factor_order')}",
        f"error_order: {best_model.get('error_order')}",
        f"cov: {best_model.get('error_cov_type')}",
        f"AIC: {best_model.get('aic')}",
        f"converged: {best_model.get('converged')}",
        f"diagnostics_passed: {best_model.get('diagnostics_passed_count')}/4",
    ]

    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.02,
        0.98,
        "\n".join(text_lines),
        va="top",
        ha="left",
        fontsize=12,
        family="monospace",
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
