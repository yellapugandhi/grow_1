import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
from data import load_data
import os

# === Load API Token ===
if "__streamlit_groww_token__" in globals():
    AUTH_TOKEN = __streamlit_groww_token__
else:
    AUTH_TOKEN = st.sidebar.text_input("Enter your Groww API token", type="password")

if not AUTH_TOKEN:
    st.warning("Please enter your Groww API token in the sidebar.")
    st.stop()

# === Load Groww Client and Data ===
groww, _, _, _, df_live = load_data(AUTH_TOKEN)

# === Load Models ===
buy_model = joblib.load("models/buy_model_latest.pkl")
rr_model = joblib.load("models/rr_model_latest.pkl")

# === Navigation ===
page = st.sidebar.radio("📚 Navigation", ["📉 Live Signal", "🧠 Retrain Model", "📘 Strategy Guide"])

# === RSI for live signal ===
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# === Live Signal Prediction ===
def live_predict():
    st.title("📈 Live Signal Predictor")
    instruments = pd.read_csv("instruments.csv")

    search = st.text_input("🔍 Search Instrument (NSE)")
    if not search:
        st.info("Enter a symbol like 'TCS', 'RELIANCE', 'HDFCBANK'")
        st.stop()

    filtered = instruments[instruments["trading_symbol"].str.contains(search.upper())]
    if filtered.empty:
        st.error("No instrument found.")
        st.stop()

    selected = filtered.iloc[0]
    interval_minutes = 10
    end_time = datetime.now()
    start_time = end_time - timedelta(days=2)

    duration_minutes = int((end_time - start_time).total_seconds() / 60)
    max_candles = 5000
    if duration_minutes / interval_minutes > max_candles:
        interval_minutes = max(60, int(duration_minutes / max_candles))

    st.markdown(f"### 🕒 Last Candle: **{df_live['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S')}**")

    try:
        data = groww.get_historical_candle_data(
            trading_symbol=selected['trading_symbol'],
            exchange=groww.EXCHANGE_NSE,
            segment=groww.SEGMENT_CASH,
            start=start_time.strftime("%Y-%m-%d %H:%M:%S"),
            end=end_time.strftime("%Y-%m-%d %H:%M:%S"),
            interval_in_minutes=interval_minutes
        )
    except Exception as e:
        st.error(f"Groww API Error: {e}")
        st.stop()

    df = pd.DataFrame(data['candles'], columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df.set_index("timestamp", inplace=True)
    df["SMA_10"] = df["close"].rolling(window=10).mean()
    df["EMA_10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["Momentum"] = df["close"] - df["close"].shift(10)
    df["Volatility"] = df["close"].rolling(window=10).std()
    df["RSI"] = compute_rsi(df["close"])

    latest = df.dropna().iloc[-1]
    X_live = latest[["SMA_10", "EMA_10", "RSI", "Momentum", "Volatility"]].values.reshape(1, -1)
    buy_signal = buy_model.predict(X_live)[0]
    confidence = buy_model.predict_proba(X_live)[0][buy_signal]
    rr = rr_model.predict(X_live)[0]

    st.subheader("🔎 Live Signal Result")
    st.markdown(f"**Instrument:** `{selected['trading_symbol']}`")
    st.markdown(f"**Buy Signal:** {'✅ BUY' if buy_signal == 1 else '❌ WAIT'}")
    st.markdown(f"**Confidence:** {confidence:.2%}")
    st.markdown(f"**Estimated Risk/Reward Ratio:** `{rr:.2f}`")
    st.line_chart(df[["close", "SMA_10", "EMA_10"]].dropna())

if page == "📉 Live Signal":
    live_predict()
elif page == "🧠 Retrain Model":
    st.title("🧠 Model Training")
    st.info("Model training is triggered using `strategy_1_model.py`. Use your terminal or CI/CD to run model training.")
elif page == "📘 Strategy Guide":
    st.title("📘 Strategy Guide")
    st.markdown("""
    This tool uses technical indicators like:
    - RSI (Relative Strength Index)
    - Momentum
    - SMA/EMA (Moving Averages)
    - Volatility
    
    Trained using a balanced dataset from historical NIFTY data.
    """)
