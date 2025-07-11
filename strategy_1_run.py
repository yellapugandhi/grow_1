# === strategy_1_run.py ===

import os
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta
from strategy_1_model import train_and_save_models
from data import prepare_df, load_data

# === Sidebar Auth ===
st.sidebar.title("🔐 Groww API Auth")
token = st.sidebar.text_input("Enter your Groww API token", type="password")
if token:
    st.session_state["auth_token"] = token

# === Sidebar Navigation ===
st.sidebar.markdown("### 🧭 Navigation")
page = st.sidebar.radio("", ["📈 Live Signal", "🧠 Retrain Model", "📘 Strategy Guide"])

# === Main Logic ===
if "auth_token" not in st.session_state:
    st.warning("Please enter your Groww API token in the sidebar.")
    st.stop()

auth_token = st.session_state["auth_token"]

# === Load Groww Client and Instrument Data ===
groww, _, _, _, _ = load_data(auth_token)

# === Model Paths ===
model_dir = "models"
buy_model_path = os.path.join(model_dir, "buy_model_latest.pkl")
rr_model_path = os.path.join(model_dir, "rr_model_latest.pkl")

# === Load or Train Models ===
if os.path.exists(buy_model_path) and os.path.exists(rr_model_path):
    buy_model = joblib.load(buy_model_path)
    rr_model = joblib.load(rr_model_path)
else:
    buy_model, rr_model = train_and_save_models(auth_token)

# === Feature Calculation for Live Data ===
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def live_predict():
    st.header("📈 Live Trading Signal")

    selected = groww.instruments[groww.instruments["exchange"] == "NSE"].iloc[0]
    symbol = selected["trading_symbol"]

    end = datetime.now()
    start = end - timedelta(days=1)
    interval_minutes = 10

    data = groww.get_historical_candle_data(
        trading_symbol=symbol,
        exchange=groww.EXCHANGE_NSE,
        segment=groww.SEGMENT_CASH,
        start=start.strftime("%Y-%m-%d %H:%M:%S"),
        end=end.strftime("%Y-%m-%d %H:%M:%S"),
        interval_in_minutes=interval_minutes
    )

    df = prepare_df(data)
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['Momentum'] = df['close'] - df['close'].shift(10)
    df['Volatility'] = df['close'].rolling(window=10).std()
    df['RSI'] = compute_rsi(df['close'])

    df.dropna(inplace=True)

    features = ['SMA_10', 'EMA_10', 'RSI', 'Momentum', 'Volatility']
    X_live = df[features].iloc[-1:]

    buy_pred = buy_model.predict(X_live)[0]
    rr_pred = rr_model.predict(X_live)[0]

    st.markdown(f"### 🕔 Last Candle: **{df['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S')}**")
    st.success(f"🟢 Buy Signal: **{'Yes' if buy_pred == 1 else 'No'}**")
    st.info(f"🎯 Estimated Risk/Reward: **{rr_pred:.2f}**")

# === Pages ===
if page == "📈 Live Signal":
    live_predict()
elif page == "🧠 Retrain Model":
    st.header("🧠 Retrain Model")
    buy_model, rr_model = train_and_save_models(auth_token)
    st.success("✅ Model retrained and saved.")
elif page == "📘 Strategy Guide":
    st.header("📘 Strategy Guide")
    st.markdown("""
    - **SMA/EMA**: 10-period simple and exponential moving averages.
    - **RSI**: Relative Strength Index (14-period)
    - **Momentum**: Price difference over 10 candles.
    - **Volatility**: Standard deviation of last 10 closes.
    - **Buy Signal**: Generated based on momentum, RSI, Bollinger Bands, MACD.
    """)
