import joblib
import pandas as pd

# === Load trained models ===
buy_model = joblib.load("models/buy_model_latest.pkl")
rr_model = joblib.load("models/rr_model_latest.pkl")

# === RSI function for use in Streamlit ===
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
