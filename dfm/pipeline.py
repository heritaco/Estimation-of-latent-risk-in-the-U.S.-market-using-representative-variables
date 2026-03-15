from __future__ import annotations

import json
import math
import re
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

try:
    from trend.data import SeriesMeta as TrendSeriesMeta
    from trend.data import from_yahoo_series as trend_from_yahoo_series
except Exception:
    trend_from_yahoo_series = None

    @dataclass(frozen=True)
    class TrendSeriesMeta:
        name: str
        start: Optional[str] = None
        end: Optional[str] = None
        use_log: bool = True
        freq: Optional[str] = None


START_DATE = "2000-01-01"
END_DATE = None
MIN_OBS = 750
MAX_HTTP_ATTEMPTS = 4
MAXITER = 400
BUSINESS_DAY_FREQ = "B"
FORWARD_FILL_LIMIT = 5
LOG_FLOOR_MIN = 1e-8
LB_LAG = 10
ARCH_LAG = 10
MIN_DIAGNOSTIC_OBS = 30

DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"

SERIES_SPECS = [
    {"name": "VIX", "source": "yahoo", "ticker": "^VIX", "kind": "positive", "required": True},
    {"name": "WTI", "source": "yahoo", "ticker": "CL=F", "kind": "positive", "required": True},
    {"name": "GOLD", "source": "yahoo", "ticker": "GLD", "kind": "positive", "required": True},
    {"name": "DXY", "source": "yahoo", "ticker": "DX-Y.NYB", "kind": "positive", "required": True},
]

MODEL_GRID = [
    {"k_factors": k_factors, "factor_order": factor_order, "error_order": error_order, "error_cov_type": error_cov_type}
    for k_factors in [1, 2]
    for factor_order in [1, 2]
    for error_order in [0, 1]
    for error_cov_type in ["diagonal", "unstructured"]
]


@dataclass(frozen=True)
class SeriesFetchResult:
    name: str
    source: str
    ticker: str
    cache_path: str
    loaded_from_cache: bool
    fallback_used: bool
    n_obs: int
    start: Optional[str]
    end: Optional[str]
    series_meta: Dict[str, Any]
    notes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelDiagnosticSummary:
    k_factors: int
    factor_order: int
    error_order: int
    error_cov_type: str
    fit_success: bool
    converged: bool
    aic: float
    bic: float
    llf: float
    pass_ljung_box: bool
    pass_jarque_bera: bool
    pass_arch: bool
    diagnostics_passed_count: int
    overall_pass: bool
    used_columns: List[str]
    fit_error: Optional[str] = None


@dataclass
class RunMetadata:
    started_at: str
    completed_at: Optional[str] = None
    stage: str = "initializing"
    series_specs: List[Dict[str, Any]] = field(default_factory=list)
    downloads: List[Dict[str, Any]] = field(default_factory=list)
    panel_checks: Dict[str, Any] = field(default_factory=dict)
    transform_checks: Dict[str, Any] = field(default_factory=dict)
    flooring_actions: List[Dict[str, Any]] = field(default_factory=list)
    best_model: Dict[str, Any] = field(default_factory=dict)
    best_model_selection_reason: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    last_attempted_model: Optional[Dict[str, Any]] = None


@dataclass
class CrashReport:
    error_type: str
    error_message: str
    traceback: str
    stage: str
    python_version: str
    crashed_at: str
    last_attempted_model: Optional[Dict[str, Any]]
    meta_partial: Dict[str, Any]


class ProgressLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a", encoding="utf-8", buffering=1)

    def log(self, message: str) -> None:
        line = f"{utc_now()} | {message}"
        print(line, flush=True)
        self.handle.write(line + "\n")

    def close(self) -> None:
        self.handle.close()


LOGGER: Optional[ProgressLogger] = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def log(message: str) -> None:
    if LOGGER is None:
        print(message, flush=True)
    else:
        LOGGER.log(message)


def log_json(label: str, payload: Dict[str, Any]) -> None:
    log(f"{label} {json.dumps(payload, indent=2, ensure_ascii=False, default=str)}")


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def cache_path_for(source: str, ticker: str) -> Path:
    return DATA_DIR / f"{source}__{safe_name(ticker)}.csv"


def make_series_meta(series: pd.Series, name: str, use_log: bool = True) -> TrendSeriesMeta:
    return TrendSeriesMeta(
        name=name,
        start=str(series.index.min().date()) if len(series) else None,
        end=str(series.index.max().date()) if len(series) else None,
        use_log=use_log,
        freq=BUSINESS_DAY_FREQ,
    )


def to_relative_string(path: Path) -> str:
    return path.relative_to(ROOT_DIR).as_posix()


def initialize_artifact_paths() -> Dict[str, Path]:
    return {
        "latent_factor": OUTPUT_DIR / "factor_latente_final.csv",
        "panel_original": OUTPUT_DIR / "panel_original_niveles.csv",
        "panel_transformed": OUTPUT_DIR / "panel_transformado_usado.csv",
        "grid": OUTPUT_DIR / "dfm_diagnostics_grid.csv",
        "summary": OUTPUT_DIR / "dfm_summary.txt",
        "metadata": OUTPUT_DIR / "dfm_run_metadata.json",
        "best_model_pickle": OUTPUT_DIR / "dfm_best_model.pkl",
        "log": OUTPUT_DIR / "dfm_run.log",
        "crash": OUTPUT_DIR / "crash_report.json",
    }


def clean_datetime_index(values: Sequence[Any]) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(pd.to_datetime(values, errors="coerce"))
    if index.tz is not None:
        index = index.tz_convert(None)
    return index


def write_series_csv(series: pd.Series, path: Path) -> None:
    output = series.dropna().sort_index().rename(series.name or "value").to_frame()
    output.index.name = "Date"
    output.to_csv(path)


def robust_load_series(path: Path, name: str) -> pd.Series:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Empty cache file: {path}")

    lower_map = {str(column).lower(): column for column in frame.columns}
    date_col = lower_map.get("date") or lower_map.get("datetime") or frame.columns[0]
    candidate_value_columns = [
        lower_map.get(name.lower()),
        lower_map.get("value"),
        lower_map.get("close"),
        lower_map.get("adj close"),
        lower_map.get("adj_close"),
        lower_map.get("price"),
    ]
    value_col = next((column for column in candidate_value_columns if column is not None), None)
    if value_col is None:
        non_date_columns = [column for column in frame.columns if column != date_col]
        if not non_date_columns:
            raise ValueError(f"No value column found in cache file: {path}")
        value_col = non_date_columns[0]

    index = clean_datetime_index(frame[date_col])
    values = pd.to_numeric(frame[value_col], errors="coerce")
    series = pd.Series(values.to_numpy(), index=index, name=name)
    series = series[~series.index.isna()].sort_index()
    series = series[~series.index.duplicated(keep="last")].dropna()
    if series.empty:
        raise ValueError(f"Parsed cache is empty: {path}")
    return series


def download_yahoo_one_by_one(ticker: str, start: str = START_DATE, end: Optional[str] = END_DATE) -> pd.Series:
    if trend_from_yahoo_series is None:
        raise ImportError("trend.data.from_yahoo_series is not available.")

    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_HTTP_ATTEMPTS + 1):
        try:
            log(f"[yahoo] downloading {ticker} via trend.data.from_yahoo_series (attempt {attempt}/{MAX_HTTP_ATTEMPTS})")
            series, _ = trend_from_yahoo_series(
                ticker=ticker,
                start=start,
                end=end,
                use_log=False,
            )
            series = pd.to_numeric(series, errors="coerce").dropna()
            series.index = clean_datetime_index(series.index)
            series = series.sort_index()
            series = series[~series.index.duplicated(keep="last")]
            if series.empty:
                raise RuntimeError(f"Yahoo parsed series is empty for {ticker}")
            series.name = ticker
            return series
        except Exception as exc:
            last_error = exc
            if attempt < MAX_HTTP_ATTEMPTS:
                wait_seconds = min(2 ** attempt, 8)
                log(f"[yahoo] {ticker} failed on attempt {attempt}: {exc}. retrying in {wait_seconds}s")
                time.sleep(wait_seconds)
    raise RuntimeError(f"Yahoo failed for {ticker}: {last_error}") from last_error
def fetch_series_from_source(spec: Dict[str, Any]) -> pd.Series:
    if spec["source"] != "yahoo":
        raise ValueError(f"Unsupported source: {spec['source']}")
    return download_yahoo_one_by_one(spec["ticker"], start=START_DATE, end=END_DATE)


def get_or_download_series(spec: Dict[str, Any]) -> Tuple[pd.Series, SeriesFetchResult]:
    cache_path = cache_path_for(spec["source"], spec["ticker"])
    if cache_path.exists():
        try:
            series = robust_load_series(cache_path, spec["name"]).rename(spec["name"])
            meta = make_series_meta(series, spec["name"])
            log(f"[cache] using cached {spec['name']} from {to_relative_string(cache_path)} with {len(series)} obs")
            return series, SeriesFetchResult(
                name=spec["name"],
                source=spec["source"],
                ticker=spec["ticker"],
                cache_path=to_relative_string(cache_path),
                loaded_from_cache=True,
                fallback_used=False,
                n_obs=int(len(series)),
                start=meta.start,
                end=meta.end,
                series_meta=asdict(meta),
            )
        except Exception as exc:
            log(f"[cache] invalid cache for {spec['name']} at {to_relative_string(cache_path)}: {exc}. redownloading")

    series = fetch_series_from_source(spec).rename(spec["name"])
    write_series_csv(series, cache_path)
    meta = make_series_meta(series, spec["name"])
    log(f"[cache] saved {spec['name']} to {to_relative_string(cache_path)}")
    return series, SeriesFetchResult(
        name=spec["name"],
        source=spec["source"],
        ticker=spec["ticker"],
        cache_path=to_relative_string(cache_path),
        loaded_from_cache=False,
        fallback_used=False,
        n_obs=int(len(series)),
        start=meta.start,
        end=meta.end,
        series_meta=asdict(meta),
    )


def series_snapshot(series: pd.Series) -> Dict[str, Any]:
    return {
        "n_obs": int(len(series)),
        "start": str(series.index.min().date()) if len(series) else None,
        "end": str(series.index.max().date()) if len(series) else None,
        "min": float(series.min()) if len(series) else math.nan,
        "max": float(series.max()) if len(series) else math.nan,
        "last": float(series.iloc[-1]) if len(series) else math.nan,
    }


def panel_snapshot(frame: pd.DataFrame) -> Dict[str, Any]:
    constant_columns = [column for column in frame.columns if frame[column].nunique(dropna=True) <= 1]
    return {
        "shape": [int(frame.shape[0]), int(frame.shape[1])],
        "start": str(frame.index.min().date()) if len(frame) else None,
        "end": str(frame.index.max().date()) if len(frame) else None,
        "duplicate_dates": int(frame.index.duplicated().sum()),
        "missing_counts": {column: int(frame[column].isna().sum()) for column in frame.columns},
        "nonpositive_counts": {column: int((frame[column] <= 0).sum()) for column in frame.columns},
        "constant_columns": constant_columns,
    }


def build_business_day_panel(series_items: Sequence[pd.Series]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    raw = pd.concat(series_items, axis=1).sort_index()
    raw = raw[~raw.index.duplicated(keep="last")]
    pre_alignment = panel_snapshot(raw)
    aligned = raw.asfreq(BUSINESS_DAY_FREQ).ffill(limit=FORWARD_FILL_LIMIT)
    post_alignment_before_drop = panel_snapshot(aligned)
    aligned = aligned.dropna(how="any")
    final_snapshot = panel_snapshot(aligned)
    if aligned.empty:
        raise RuntimeError("Merged business-day panel is empty after dropping missing rows")
    if aligned.shape[0] < MIN_OBS:
        raise RuntimeError(f"Merged business-day panel has only {aligned.shape[0]} rows; need at least {MIN_OBS}")
    if final_snapshot["constant_columns"]:
        raise RuntimeError(f"Constant columns after alignment: {final_snapshot['constant_columns']}")
    return aligned, {
        "pre_alignment": pre_alignment,
        "post_alignment_before_drop": post_alignment_before_drop,
        "final_panel": final_snapshot,
    }


def transformed_snapshot(frame: pd.DataFrame) -> Dict[str, Any]:
    array = frame.to_numpy(dtype=float)
    return {
        "shape": [int(frame.shape[0]), int(frame.shape[1])],
        "nan_total": int(np.isnan(array).sum()),
        "inf_total": int(np.isinf(array).sum()),
        "zero_var_columns": [column for column in frame.columns if np.isclose(frame[column].std(ddof=0), 0.0)],
        "mean_abs_max": float(np.abs(frame.mean()).max()) if len(frame) else math.nan,
        "std_dev_from_1_max": float(np.abs(frame.std(ddof=0) - 1.0).max()) if len(frame) else math.nan,
    }


def apply_safe_log_and_standardize(panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]:
    adjusted = panel.copy()
    flooring_actions: List[Dict[str, Any]] = []

    for column in adjusted.columns:
        series = pd.to_numeric(adjusted[column], errors="coerce")
        if series.isna().any():
            raise RuntimeError(f"Series {column} contains NaN before log transform")
        nonpositive_mask = series <= 0
        if nonpositive_mask.any():
            positive_values = series[series > 0]
            if positive_values.empty:
                raise RuntimeError(f"Series {column} has no positive observations; cannot apply safe log transform")
            floor_value = max(float(positive_values.min()) * 0.5, LOG_FLOOR_MIN)
            count = int(nonpositive_mask.sum())
            adjusted.loc[nonpositive_mask, column] = floor_value
            action = {
                "series": column,
                "replaced_count": count,
                "floor_value": floor_value,
            }
            flooring_actions.append(action)
            log(f"[transform] {column}: replaced {count} non-positive values with floor {floor_value:.10g}")

    logged = np.log(adjusted)
    standardized = (logged - logged.mean()) / logged.std(ddof=0)
    checks = transformed_snapshot(standardized)
    if checks["nan_total"] != 0 or checks["inf_total"] != 0:
        raise RuntimeError(f"Transformed panel contains invalid values: {checks}")
    if checks["zero_var_columns"]:
        raise RuntimeError(f"Transformed panel contains zero-variance columns: {checks['zero_var_columns']}")
    if standardized.shape[0] < MIN_OBS:
        raise RuntimeError(f"Transformed panel has only {standardized.shape[0]} rows; need at least {MIN_OBS}")
    return logged, standardized, flooring_actions, checks


def result_to_residual_frame(result: Any, columns: Sequence[str], index: pd.Index) -> pd.DataFrame:
    residuals = np.asarray(result.resid)
    expected_shape = (len(index), len(columns))
    if residuals.shape == expected_shape:
        residual_frame = pd.DataFrame(residuals, index=index, columns=columns)
    elif residuals.T.shape == expected_shape:
        residual_frame = pd.DataFrame(residuals.T, index=index, columns=columns)
    else:
        raise ValueError(f"Unexpected residual shape {residuals.shape}; expected {expected_shape} or {expected_shape[::-1]}")
    return residual_frame


def residual_diagnostics(result: Any, x_data: pd.DataFrame) -> Dict[str, Any]:
    residual_frame = result_to_residual_frame(result, list(x_data.columns), x_data.index)
    details: Dict[str, Dict[str, Any]] = {}
    pass_ljung_box = True
    pass_jarque_bera = True
    pass_arch = True

    for column in residual_frame.columns:
        series = residual_frame[column].dropna()
        detail: Dict[str, Any] = {}
        if len(series) < max(MIN_DIAGNOSTIC_OBS, LB_LAG + 1, ARCH_LAG + 1):
            detail["error"] = f"too_short:{len(series)}"
            detail["ljung_box_pvalue"] = math.nan
            detail["jarque_bera_pvalue"] = math.nan
            detail["arch_pvalue"] = math.nan
            detail["pass_ljung_box"] = False
            detail["pass_jarque_bera"] = False
            detail["pass_arch"] = False
        else:
            try:
                lb_pvalue = float(acorr_ljungbox(series, lags=[LB_LAG], return_df=True)["lb_pvalue"].iloc[0])
            except Exception as exc:
                lb_pvalue = math.nan
                detail["ljung_box_error"] = str(exc)
            try:
                jb_result = stats.jarque_bera(series)
                jb_pvalue = float(jb_result.pvalue if hasattr(jb_result, "pvalue") else jb_result[1])
            except Exception as exc:
                jb_pvalue = math.nan
                detail["jarque_bera_error"] = str(exc)
            try:
                arch_pvalue = float(het_arch(series, nlags=ARCH_LAG)[1])
            except Exception as exc:
                arch_pvalue = math.nan
                detail["arch_error"] = str(exc)

            detail["ljung_box_pvalue"] = lb_pvalue
            detail["jarque_bera_pvalue"] = jb_pvalue
            detail["arch_pvalue"] = arch_pvalue
            detail["pass_ljung_box"] = bool(math.isfinite(lb_pvalue) and lb_pvalue > 0.05)
            detail["pass_jarque_bera"] = bool(math.isfinite(jb_pvalue) and jb_pvalue > 0.05)
            detail["pass_arch"] = bool(math.isfinite(arch_pvalue) and arch_pvalue > 0.05)

        pass_ljung_box = pass_ljung_box and detail["pass_ljung_box"]
        pass_jarque_bera = pass_jarque_bera and detail["pass_jarque_bera"]
        pass_arch = pass_arch and detail["pass_arch"]
        details[column] = detail

    return {
        "pass_ljung_box": pass_ljung_box,
        "pass_jarque_bera": pass_jarque_bera,
        "pass_arch": pass_arch,
        "detail": details,
    }


def fit_one_model(x_data: pd.DataFrame, spec: Dict[str, Any]) -> Tuple[Any, ModelDiagnosticSummary, Dict[str, Any]]:
    model = DynamicFactor(
        endog=x_data,
        k_factors=spec["k_factors"],
        factor_order=spec["factor_order"],
        error_order=spec["error_order"],
        error_cov_type=spec["error_cov_type"],
    )
    log(
        "[model] fitting "
        f"k={spec['k_factors']} factor_order={spec['factor_order']} error_order={spec['error_order']} "
        f"error_cov_type={spec['error_cov_type']}"
    )
    result = model.fit(maxiter=MAXITER, disp=False)
    converged = bool(getattr(result, "mle_retvals", {}).get("converged", False))
    diagnostics = residual_diagnostics(result, x_data)
    passed_count = int(converged) + int(diagnostics["pass_ljung_box"]) + int(diagnostics["pass_jarque_bera"]) + int(diagnostics["pass_arch"])
    summary = ModelDiagnosticSummary(
        k_factors=spec["k_factors"],
        factor_order=spec["factor_order"],
        error_order=spec["error_order"],
        error_cov_type=spec["error_cov_type"],
        fit_success=True,
        converged=converged,
        aic=float(result.aic),
        bic=float(result.bic),
        llf=float(result.llf),
        pass_ljung_box=diagnostics["pass_ljung_box"],
        pass_jarque_bera=diagnostics["pass_jarque_bera"],
        pass_arch=diagnostics["pass_arch"],
        diagnostics_passed_count=passed_count,
        overall_pass=bool(converged and diagnostics["pass_ljung_box"] and diagnostics["pass_jarque_bera"] and diagnostics["pass_arch"]),
        used_columns=list(x_data.columns),
    )
    return result, summary, diagnostics


def flatten_diagnostics_for_grid(diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for series_name, detail in diagnostics.get("detail", {}).items():
        prefix = f"series_{series_name}"
        flattened[f"{prefix}_ljung_box_pvalue"] = detail.get("ljung_box_pvalue")
        flattened[f"{prefix}_jarque_bera_pvalue"] = detail.get("jarque_bera_pvalue")
        flattened[f"{prefix}_arch_pvalue"] = detail.get("arch_pvalue")
        flattened[f"{prefix}_pass_ljung_box"] = detail.get("pass_ljung_box")
        flattened[f"{prefix}_pass_jarque_bera"] = detail.get("pass_jarque_bera")
        flattened[f"{prefix}_pass_arch"] = detail.get("pass_arch")
        if "error" in detail:
            flattened[f"{prefix}_error"] = detail["error"]
    return flattened


def simplicity_key(summary: ModelDiagnosticSummary) -> Tuple[int, int, int, int]:
    return (
        summary.k_factors,
        summary.factor_order,
        summary.error_order,
        0 if summary.error_cov_type == "diagonal" else 1,
    )


def choose_best_model(summaries: Sequence[ModelDiagnosticSummary]) -> Tuple[ModelDiagnosticSummary, str]:
    successful = [summary for summary in summaries if summary.fit_success]
    if not successful:
        raise RuntimeError("No Dynamic Factor Model specification could be fitted successfully")

    fully_passing = [summary for summary in successful if summary.overall_pass]
    if fully_passing:
        return min(fully_passing, key=simplicity_key), (
            "Selected the simplest model that passed convergence, Ljung-Box, Jarque-Bera, and ARCH diagnostics."
        )

    def fallback_key(summary: ModelDiagnosticSummary) -> Tuple[float, float, Tuple[int, int, int, int]]:
        aic_value = summary.aic if math.isfinite(summary.aic) else math.inf
        return (-summary.diagnostics_passed_count, aic_value, simplicity_key(summary))

    return min(successful, key=fallback_key), (
        "No model passed all diagnostics; selected the model with the highest diagnostics count, then lowest AIC, then simplicity."
    )


def run_model_grid(x_data: pd.DataFrame, metadata: RunMetadata) -> Tuple[Any, ModelDiagnosticSummary, Dict[str, Any], pd.DataFrame, str]:
    grid_rows: List[Dict[str, Any]] = []
    summaries: List[ModelDiagnosticSummary] = []
    fitted_results: Dict[Tuple[int, int, int, str], Tuple[Any, Dict[str, Any]]] = {}

    total_models = len(MODEL_GRID)
    for position, spec in enumerate(MODEL_GRID, start=1):
        metadata.stage = f"fitting_model_{position}_of_{total_models}"
        metadata.last_attempted_model = dict(spec)
        log(f"[grid] model {position}/{total_models}: {spec}")
        try:
            result, summary, diagnostics = fit_one_model(x_data, spec)
            summaries.append(summary)
            fitted_results[(summary.k_factors, summary.factor_order, summary.error_order, summary.error_cov_type)] = (result, diagnostics)
            row = {**asdict(summary), **flatten_diagnostics_for_grid(diagnostics)}
            grid_rows.append(row)
            log(
                "[grid] completed "
                f"{spec} | converged={summary.converged} lb={summary.pass_ljung_box} "
                f"jb={summary.pass_jarque_bera} arch={summary.pass_arch} aic={summary.aic:.3f}"
            )
        except Exception as exc:
            log(f"[grid] failed {spec}: {exc}")
            summary = ModelDiagnosticSummary(
                k_factors=spec["k_factors"],
                factor_order=spec["factor_order"],
                error_order=spec["error_order"],
                error_cov_type=spec["error_cov_type"],
                fit_success=False,
                converged=False,
                aic=math.nan,
                bic=math.nan,
                llf=math.nan,
                pass_ljung_box=False,
                pass_jarque_bera=False,
                pass_arch=False,
                diagnostics_passed_count=0,
                overall_pass=False,
                used_columns=list(x_data.columns),
                fit_error=str(exc),
            )
            summaries.append(summary)
            grid_rows.append(asdict(summary))

    best_summary, selection_reason = choose_best_model(summaries)
    best_key = (
        best_summary.k_factors,
        best_summary.factor_order,
        best_summary.error_order,
        best_summary.error_cov_type,
    )
    best_result, best_diagnostics = fitted_results[best_key]
    grid_frame = pd.DataFrame(grid_rows)
    return best_result, best_summary, best_diagnostics, grid_frame, selection_reason


def extract_oriented_factor(result: Any, x_data: pd.DataFrame, logged_panel: pd.DataFrame) -> pd.Series:
    filtered_factors = np.asarray(result.factors.filtered)
    if filtered_factors.ndim == 1:
        factor_values = filtered_factors
    else:
        factor_values = filtered_factors[0]
    if len(factor_values) != len(x_data.index):
        raise RuntimeError(f"Unexpected factor length {len(factor_values)}; expected {len(x_data.index)}")

    latent_factor = pd.Series(factor_values, index=x_data.index, name="latent_factor")
    if "VIX" in logged_panel.columns:
        aligned = pd.concat([latent_factor, logged_panel["VIX"].rename("log_VIX")], axis=1).dropna()
        if len(aligned) >= 20:
            correlation = aligned["latent_factor"].corr(aligned["log_VIX"])
            log(f"[factor] correlation with log(VIX) before orientation: {correlation:.6f}")
            if pd.notna(correlation) and correlation < 0:
                latent_factor = -latent_factor
                log("[factor] latent factor sign flipped to align positively with log(VIX)")
    return latent_factor


def write_json(payload: Dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)


def write_summary_text(
    path: Path,
    best_result: Any,
    best_summary: ModelDiagnosticSummary,
    diagnostics: Dict[str, Any],
    selection_reason: str,
    grid_frame: pd.DataFrame,
) -> None:
    ranking_columns = [
        column
        for column in [
            "k_factors",
            "factor_order",
            "error_order",
            "error_cov_type",
            "fit_success",
            "converged",
            "diagnostics_passed_count",
            "overall_pass",
            "aic",
            "bic",
            "llf",
            "fit_error",
        ]
        if column in grid_frame.columns
    ]
    sortable = grid_frame.copy()
    if "diagnostics_passed_count" in sortable.columns:
        sortable["_sort_diag"] = -pd.to_numeric(sortable["diagnostics_passed_count"], errors="coerce").fillna(-999)
    else:
        sortable["_sort_diag"] = 999
    if "aic" in sortable.columns:
        sortable["_sort_aic"] = pd.to_numeric(sortable["aic"], errors="coerce").fillna(np.inf)
    else:
        sortable["_sort_aic"] = np.inf
    if "error_cov_type" in sortable.columns:
        sortable["_sort_cov"] = sortable["error_cov_type"].map({"diagonal": 0, "unstructured": 1}).fillna(99)
    else:
        sortable["_sort_cov"] = 99
    top_rows = sortable.sort_values(
        by=[
            "_sort_diag",
            "_sort_aic",
            "k_factors",
            "factor_order",
            "error_order",
            "_sort_cov",
        ],
        ascending=[True, True, True, True, True, True],
    )[ranking_columns].head(5)

    with path.open("w", encoding="utf-8") as handle:
        handle.write("Robust DFM latent-series pipeline\n")
        handle.write("=" * 40 + "\n\n")
        handle.write("Best model\n")
        handle.write(json.dumps(asdict(best_summary), indent=2, ensure_ascii=False, default=str))
        handle.write("\n\nSelection rule outcome\n")
        handle.write(selection_reason + "\n\n")
        handle.write("Top ranked model rows\n")
        handle.write(top_rows.to_string(index=False))
        handle.write("\n\nBest model statsmodels summary\n")
        handle.write(str(best_result.summary()))
        handle.write("\n\nResidual diagnostics\n")
        handle.write(json.dumps(diagnostics, indent=2, ensure_ascii=False, default=str))


def remove_stale_crash_report(path: Path) -> None:
    if path.exists():
        path.unlink()


def main() -> None:
    global LOGGER

    ensure_dirs()
    artifacts = initialize_artifact_paths()
    LOGGER = ProgressLogger(artifacts["log"])

    metadata = RunMetadata(
        started_at=utc_now(),
        series_specs=SERIES_SPECS,
        artifacts={name: to_relative_string(path) for name, path in artifacts.items()},
    )

    try:
        log("[start] robust DFM latent-series pipeline starting")
        log(f"[start] root directory: {ROOT_DIR}")
        log(f"[start] data directory: {DATA_DIR}")
        log(f"[start] output directory: {OUTPUT_DIR}")

        metadata.stage = "loading_series"
        loaded_series: List[pd.Series] = []
        for position, spec in enumerate(SERIES_SPECS, start=1):
            log(f"[data] loading series {position}/{len(SERIES_SPECS)}: {spec['name']} from {spec['source']}:{spec['ticker']}")
            series, result = get_or_download_series(spec)
            metadata.downloads.append(asdict(result))

            log_json(f"[data] snapshot for {spec['name']}:", series_snapshot(series))
            loaded_series.append(series.rename(spec["name"]))

        metadata.stage = "building_panel"
        raw_panel, panel_checks = build_business_day_panel(loaded_series)
        metadata.panel_checks = panel_checks
        log_json("[panel] checks:", panel_checks)

        raw_panel.to_csv(artifacts["panel_original"], index_label="Date")
        log(f"[output] wrote {to_relative_string(artifacts['panel_original'])}")

        metadata.stage = "transforming_panel"
        logged_panel, transformed_panel, flooring_actions, transform_checks = apply_safe_log_and_standardize(raw_panel)
        metadata.flooring_actions = flooring_actions
        metadata.transform_checks = transform_checks
        log_json("[transform] checks:", transform_checks)
        if flooring_actions:
            log_json("[transform] flooring actions:", {"actions": flooring_actions})
        else:
            log("[transform] no flooring actions were required")

        transformed_panel.to_csv(artifacts["panel_transformed"], index_label="Date")
        log(f"[output] wrote {to_relative_string(artifacts['panel_transformed'])}")

        metadata.stage = "running_model_grid"
        best_result, best_summary, best_diagnostics, grid_frame, selection_reason = run_model_grid(transformed_panel, metadata)
        grid_frame.to_csv(artifacts["grid"], index=False)
        log(f"[output] wrote {to_relative_string(artifacts['grid'])}")

        metadata.stage = "extracting_factor"
        latent_factor = extract_oriented_factor(best_result, transformed_panel, logged_panel)
        latent_factor.to_frame().to_csv(artifacts["latent_factor"], index_label="Date")
        log_json("[factor] snapshot:", series_snapshot(latent_factor))
        log(f"[output] wrote {to_relative_string(artifacts['latent_factor'])}")

        metadata.stage = "writing_artifacts"
        best_result.save(str(artifacts["best_model_pickle"]))
        log(f"[output] wrote {to_relative_string(artifacts['best_model_pickle'])}")

        write_summary_text(artifacts["summary"], best_result, best_summary, best_diagnostics, selection_reason, grid_frame)
        log(f"[output] wrote {to_relative_string(artifacts['summary'])}")

        metadata.best_model = asdict(best_summary)
        metadata.best_model_selection_reason = selection_reason
        metadata.diagnostics = best_diagnostics
        metadata.completed_at = utc_now()
        write_json(asdict(metadata), artifacts["metadata"])
        log(f"[output] wrote {to_relative_string(artifacts['metadata'])}")

        remove_stale_crash_report(artifacts["crash"])
        log("[done] latent factor pipeline completed successfully")
    except Exception as exc:
        crash = CrashReport(
            error_type=type(exc).__name__,
            error_message=str(exc),
            traceback=traceback.format_exc(),
            stage=metadata.stage,
            python_version=sys.version,
            crashed_at=utc_now(),
            last_attempted_model=metadata.last_attempted_model,
            meta_partial=asdict(metadata),
        )
        write_json(asdict(crash), artifacts["crash"])
        log(f"[error] pipeline failed during stage '{metadata.stage}'. crash report written to {to_relative_string(artifacts['crash'])}")
        raise
    finally:
        if LOGGER is not None:
            LOGGER.close()
            LOGGER = None


if __name__ == "__main__":
    main()
