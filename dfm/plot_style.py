from __future__ import annotations

from typing import Iterable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.ticker import MaxNLocator

BACKGROUND_COLOR = "#E0E1E5"
FIGURE_SIZE = (10, 8/3)
SCATTER_FIGURE_SIZE = (7, 4.5)
PALETTE_1 = "#005869"
PALETTE_2 = "#074259"
PALETTE_3 = "#142b48"
PALETTE_4 = "#2c1f40"
PALETTE_5 = "#4d2045"
PALETTE_6 = "#5f0e39"
PALETTE_7 = "#6f002d"

plt.rcParams["font.family"] = "EB Garamond"
plt.rcParams["font.serif"] = ["EB Garamond", "serif"]
plt.rcParams["axes.prop_cycle"] = cycler(
    color=[PALETTE_1, PALETTE_2, PALETTE_3, PALETTE_4, PALETTE_5, PALETTE_6, PALETTE_7]
)


def style_figure(fig) -> None:
    fig.patch.set_facecolor(BACKGROUND_COLOR)


def style_axis(ax, *, time_axis: bool = False, xbins: int = 6, ybins: int = 6) -> None:
    ax.set_facecolor(BACKGROUND_COLOR)
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
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    ax.title.set_color("black")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=ybins))

    if time_axis:
        locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=xbins))


def style_secondary_axis(ax, *, ybins: int = 6) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(
        axis="y",
        which="major",
        direction="out",
        length=6,
        width=1,
        colors="black",
        labelright=True,
    )
    ax.yaxis.label.set_color("black")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=ybins))


def style_grid(ax, *, axis: str = "both") -> None:
    ax.grid(axis=axis, alpha=0.25)


def style_legend(ax, *, ncol: int = 1, handles: Iterable | None = None, labels: Iterable[str] | None = None) -> None:
    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()
    handles = list(handles)
    labels = list(labels)
    if not handles:
        return
    ax.legend(handles, labels, loc="upper left", frameon=False, ncol=ncol)


def style_text_box(
    ax,
    text: str,
    *,
    x: float = 0.02,
    y: float = 0.98,
    va: str = "top",
    ha: str = "left",
) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        va=va,
        ha=ha,
        bbox={"facecolor": BACKGROUND_COLOR, "alpha": 0.9, "edgecolor": "black"},
    )
