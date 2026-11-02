"""
PART 4 - NUMPY AND TIME SERIES FORECAST (Statsmodels)
Run AFTER data_statistics.py. Produces p2_15_forecast_arima.csv.
Syllabus: Unit 3.1 (NumPy), Unit 5.2 (Statsmodels: stationarity, smoothing, decomposition, ARIMA).
"""

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

import pmkisan_utils as pu          # our own module (keep it in the same folder)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)


# =============================================================================
# UNIT 3.1  NUMPY
# =============================================================================
def numpy_section(nat):
    pu.heading("Unit 3.1 NumPy: national series as arrays")
    total = nat["Beneficiaries"].to_numpy() / 1e6            # 1-D array (millions)
    print("1-D array shape:", total.shape, "| first 3:", total[:3].round(1))
    print("Mean %.1f | Median %.1f | Std %.1f" % (total.mean(), np.median(total), total.std()))
    change = np.diff(total) / total[:-1] * 100               # maths on arrays
    print("Instalments with a fall (relational operator):",
          (np.where(change < 0)[0] + 2).tolist())            # np.where + comparison
    falls = change[change < 0]                               # boolean indexing
    print("Average size of a fall: %.2f%%" % falls.mean())
    grid = np.vstack([total[:-1], total[1:]])                # 2-D array: previous vs current
    print("2-D array shape:", grid.shape, "| column means:", grid.mean(axis=0)[:3].round(1))
    peak, pos = pu.peak_of(total.tolist())                   # 'array' module helper
    print(pu.line("Peak (million)", round(peak, 1)), "at instalment", pos + 1)
    print("Recursive total matches NumPy total:",
          abs(pu.total_recursive(total.tolist()) - total.sum()) < 1e-6)


# =============================================================================
# UNIT 5.2  STATSMODELS TIME SERIES
# =============================================================================
def timeseries_section(nat):
    pu.heading("Unit 5.2 Statsmodels time series")
    idx = pd.DatetimeIndex(pu.instalment_dates(), freq="4MS")      # time series object
    ts = pd.Series(nat["Beneficiaries"].values / 1e6, index=idx, name="Beneficiaries_M")
    print(ts.head(3).round(1).to_dict())
    sub = ts["2021-01-01":]                                        # subset by date
    print("Subset from 2021:", len(sub), "instalments")

    # stationarity: ADF test, then differencing
    p0 = adfuller(ts, maxlag=2, autolag=None)[1]
    diff1 = ts.diff().dropna()
    p1 = adfuller(diff1, maxlag=2, autolag=None)[1]
    print("ADF p-value (level): %.3f -> %s" % (p0, "stationary" if p0 < 0.05 else "NOT stationary"))
    print("ADF p-value (1st difference): %.3f -> %s" % (p1, "stationary" if p1 < 0.05 else "NOT stationary"))

    # smoothing: simple and exponential weighted moving averages
    sma = ts.rolling(3).mean()
    ewma = ts.ewm(span=3).mean()
    print("Last SMA(3): %.1f | last EWMA(span 3): %.1f | actual last: %.1f"
          % (sma.iloc[-1], ewma.iloc[-1], ts.iloc[-1]))

    # seasonality: 3 instalments per year -> period = 3
    dec = seasonal_decompose(ts, model="additive", period=3)
    print("Seasonal pattern (Dec-Mar, Apr-Jul, Aug-Nov):", dec.seasonal.iloc[:3].round(2).tolist())
    print("Seasonal differencing (lag 3), last value: %.1f" % ts.diff(3).iloc[-1])

    # ARIMA: small grid chosen by AIC, then forecast 3 instalments
    best = None
    for order in [(0, 1, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1)]:
        fit = ARIMA(ts, order=order).fit()
        print("  ARIMA%s  AIC = %.1f" % (order, fit.aic))
        if best is None or fit.aic < best[1].aic:
            best = (order, fit)
    order, fit = best
    fc = fit.get_forecast(3)
    out = pd.DataFrame({"Instalment": [19, 20, 21],
                        "Forecast_million": fc.predicted_mean.values.round(1),
                        "Low_95": fc.conf_int().iloc[:, 0].values.round(1),
                        "High_95": fc.conf_int().iloc[:, 1].values.round(1)})
    out.to_csv("p2_15_forecast_arima.csv", index=False)
    print("Chosen model ARIMA%s. Forecast:" % (order,))
    print(out)
    print("Note: only 18 points and one big break at instalment 12, so treat the forecast as indicative.")


# =============================================================================
# MAIN
# =============================================================================
def main():
    nat = pu.one_row_per_instalment(pu.load_csv(pu.NATIONAL_FILE))
    numpy_section(nat)
    timeseries_section(nat)


if __name__ == "__main__":
    main()
