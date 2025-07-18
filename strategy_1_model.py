import pandas as pd
import joblib

# === Load trained models ===
buy_model = joblib.load("models/buy_model_latest.pkl")
rr_model = joblib.load("models/rr_model_latest.pkl")

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / (loss.replace(0, 1e-8))  # avoid division by zero
    return 100 - (100 / (1 + rs))

def compute_macd(series: pd.Series, span1: int = 12, span2: int = 26, span_signal: int = 9):
    ema1 = series.ewm(span=span1, adjust=False).mean()
    ema2 = series.ewm(span=span2, adjust=False).mean()
    macd = ema1 - ema2
    signal = macd.ewm(span=span_signal, adjust=False).mean()
    return macd, signal
