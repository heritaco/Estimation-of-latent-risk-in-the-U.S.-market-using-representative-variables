# guerrero-trend

Spectral implementation of Guerrero (2007) penalized trend with:
- arbitrary 1D series input (NumPy/Pandas)
- train/validation/test protocol
- scan of validation loss over smoothness index `s∈(0,1)` (detect **all** local minima)
- optional CLI + plotting utilities
- example notebook

## Install (editable)

```bash
pip install -e .
# optional Yahoo Finance loader:
pip install -e ".[yahoo]"
```

## Minimal usage (Python)

```python
import numpy as np
from guerrero_trend import train_val_test_split, GuerreroTrendEstimator

Z = np.log(np.arange(1, 2000) + 100 + 10*np.sin(np.arange(2000)/30))  # any 1D series
Z_train, Z_val, Z_test, split = train_val_test_split(Z, frac_train=0.6, frac_val=0.2)

est = GuerreroTrendEstimator(d=2, n_train=len(Z_train))
scan = est.scan_local_minima(Z_train, Z_val, n_grid=400)

best = scan.best()
fit = est.fit_train(Z_train, s_unit=best.s_unit)

poly_full = est.build_polynomial_full(
    t_hat_train=fit.t_hat,
    m_hat=fit.m_hat,
    n_total=len(Z),
    n_train=len(Z_train),
)
```

## CLI (optional)

```bash
guerrero-trend --ticker NVDA --start 2010-01-01 --d 2 --log --out ./out --plot
# or from a CSV:
guerrero-trend --csv path/to/data.csv --date-col Date --value-col Adj Close --log --d 2 --out ./out --plot
```

Outputs:
- `scan_d=<d>.csv` with all detected minima and validation losses
- plots (PNG) per minimum (if `--plot`)

## Notes
- `yfinance` is an *optional* dependency (install extra `.[yahoo]`).
- The core solver only depends on NumPy.


## ARIMA forecasting on the trend (optional)

Install:

```bash
pip install -e ".[arima]"
```

Usage:

```python
from guerrero_trend import GuerreroTrendEstimator

fit = est.fit_train(Z_train, s_unit=best.s_unit)
fc = est.forecast_trend_arima(fit.t_hat, h=len(Z_val), order=(2,0,2))
# fc.forecast is the ARIMA forecast of the trend
```
