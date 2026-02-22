# -*- coding: utf-8 -*-
'''
CLI entrypoint.

Examples:
  guerrero-trend --ticker NVDA --start 2010-01-01 --d 2 --log --out ./out --plot
  guerrero-trend --csv data.csv --date-col Date --value-col "Adj Close" --d 2 --log --out ./out --plot
'''
from __future__ import annotations

import argparse
import os
import json
import pandas as pd
import matplotlib.pyplot as plt

from .data import from_csv, from_yahoo
from .split import train_val_test_split
from .estimator import GuerreroTrendEstimator
from .plotting import plot_scan, plot_train_val_test_fit


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="guerrero-trend", add_help=True)

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ticker", type=str, help="Yahoo Finance ticker (requires yfinance extra).")
    src.add_argument("--csv", type=str, help="Path to CSV file.")

    p.add_argument("--start", type=str, default="2010-01-01", help="Start date (yahoo).")
    p.add_argument("--end", type=str, default=None, help="End date (yahoo).")
    p.add_argument("--date-col", type=str, default="Date", help="CSV date column (optional).")
    p.add_argument("--value-col", type=str, default="Adj Close", help="CSV value column.")
    p.add_argument("--log", action="store_true", help="Use log of the series.")

    p.add_argument("--d", type=int, nargs="+", default=[2], help="Difference order(s).")
    p.add_argument("--frac-train", type=float, default=0.6)
    p.add_argument("--frac-val", type=float, default=0.2)
    p.add_argument("--min-train", type=int, default=80)
    p.add_argument("--min-val", type=int, default=60)

    p.add_argument("--s-min", type=float, default=1e-3)
    p.add_argument("--s-max", type=float, default=0.999)
    p.add_argument("--n-grid", type=int, default=400)
    p.add_argument("--refine-iter", type=int, default=25)

    p.add_argument("--out", type=str, default="./out", help="Output directory.")
    p.add_argument("--plot", action="store_true", help="Save plots (PNG) per minimum.")
    p.add_argument("--show", action="store_true", help="Show plots interactively.")

    return p.parse_args()


def main() -> None:
    args = _parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.csv:
        Z, meta = from_csv(
            csv_path=args.csv,
            date_col=args.date_col,
            value_col=args.value_col,
            use_log=bool(args.log),
        )
    else:
        Z, meta = from_yahoo(
            ticker=args.ticker,
            start=args.start,
            end=args.end,
            use_log=bool(args.log),
        )

    Z_train, Z_val, Z_test, split = train_val_test_split(
        Z, frac_train=args.frac_train, frac_val=args.frac_val, min_train=args.min_train, min_val=args.min_val
    )

    run_meta = {
        "source": "csv" if args.csv else "yahoo",
        "meta": meta.__dict__,
        "split": split.__dict__,
        "d_list": args.d,
        "scan": {"s_min": args.s_min, "s_max": args.s_max, "n_grid": args.n_grid, "refine_iter": args.refine_iter},
    }
    with open(os.path.join(args.out, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    for d in args.d:
        est = GuerreroTrendEstimator(d=int(d), n_train=len(Z_train))
        scan = est.scan_local_minima(
            Z_train=Z_train,
            Z_val=Z_val,
            s_min=args.s_min,
            s_max=args.s_max,
            n_grid=args.n_grid,
            refine=True,
            refine_iter=args.refine_iter,
        )

        rows = [{"d": d, "s_unit": p.s_unit, "val_mse": p.val_mse} for p in scan.minima]
        pd.DataFrame(rows).to_csv(os.path.join(args.out, f"scan_d={d}.csv"), index=False)

        if args.plot or args.show:
            fig, ax = plt.subplots()
            plot_scan(scan, title=f"{meta.name} | d={d} validation scan", ax=ax)
            fig.tight_layout()
            if args.plot:
                fig.savefig(os.path.join(args.out, f"scan_d={d}.png"), dpi=160)
            if args.show:
                plt.show()
            plt.close(fig)

        if args.plot or args.show:
            for k, pnt in enumerate(scan.minima, start=1):
                fit = est.fit_train(Z_train, s_unit=pnt.s_unit)
                poly_full = est.build_polynomial_full(
                    t_hat_train=fit.t_hat,
                    m_hat=fit.m_hat,
                    n_total=len(Z),
                    n_train=len(Z_train),
                )
                title = f"{meta.name} | d={d} | s={pnt.s_unit:.4f} | val MSE={pnt.val_mse:.3e}"
                fig, ax = plt.subplots()
                plot_train_val_test_fit(
                    Z_all=Z,
                    n_train=split.n_train,
                    n_val=split.n_val,
                    t_hat_train=fit.t_hat,
                    poly_full=poly_full,
                    label=("log " if meta.use_log else "") + meta.name,
                    title=title,
                    ax=ax,
                )
                fig.tight_layout()
                if args.plot:
                    fig.savefig(os.path.join(args.out, f"fit_d={d}_k={k}_s={pnt.s_unit:.4f}.png"), dpi=160)
                if args.show:
                    plt.show()
                plt.close(fig)
