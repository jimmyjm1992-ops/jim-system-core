import numpy as np

def ema(series, period):
    series = np.asarray(series, dtype=float)
    if len(series) < period:
        return np.array([])
    alpha = 2 / (period + 1)
    out = np.zeros_like(series, dtype=float)
    out[:] = np.nan
    out[period-1] = np.mean(series[:period])
    for i in range(period, len(series)):
        out[i] = alpha * series[i] + (1 - alpha) * out[i-1]
    return out

def rsi(series, period=14):
    series = np.asarray(series, dtype=float)
    if len(series) < period + 1:
        return np.array([])
    deltas = np.diff(series)
    up = np.where(deltas > 0, deltas, 0.0)
    dn = np.where(deltas < 0, -deltas, 0.0)
    ema_up = ema(np.concatenate([[0], up]), period)
    ema_dn = ema(np.concatenate([[0], dn]), period)
    if len(ema_up) == 0 or len(ema_dn) == 0:
        return np.array([])
    rs = np.divide(ema_up, np.where(ema_dn == 0, np.nan, ema_dn))
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    series = np.asarray(series, dtype=float)
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    if len(ema_fast) == 0 or len(ema_slow) == 0:
        return np.array([]), np.array([]), np.array([])
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def donchian_high(series_high, lookback=20):
    arr = np.asarray(series_high, dtype=float)
    if len(arr) < lookback:
        return np.array([])
    out = np.zeros_like(arr, dtype=float)
    out[:] = np.nan
    for i in range(lookback-1, len(arr)):
        out[i] = np.max(arr[i-lookback+1:i+1])
    return out

def donchian_low(series_low, lookback=20):
    arr = np.asarray(series_low, dtype=float)
    if len(arr) < lookback:
        return np.array([])
    out = np.zeros_like(arr, dtype=float)
    out[:] = np.nan
    for i in range(lookback-1, len(arr)):
        out[i] = np.min(arr[i-lookback+1:i+1])
    return out
