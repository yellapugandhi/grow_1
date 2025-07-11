import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from growwapi import GrowwAPI
import joblib
import subprocess
import sys
import os
from functools import lru_cache

st.set_page_config(page_title="Trading Signal Predictor", layout="wide")

# === Groww API Auth ===
st.sidebar.title("🔐 Groww API Auth")
api_key = st.sidebar.text_input("Enter your Groww API token", type="password")

if not api_key:
    st.warning("Please enter your Groww API token in the sidebar.")
    st.stop()

# === Navigation ===
page = st.sidebar.radio("📚 Navigation", [
    "📈 Live Signal",
    "🧠 Retrain Model",
    "📘 Strategy Guide",
    "📊 Backtest",
    "🔬 Trade Lab"
])

# === Initialize Groww ===
groww = GrowwAPI(api_key)

@lru_cache(maxsize=1)
def load_instruments():
    df = pd.read_csv("instruments.csv")
    groww.instruments = df
    groww._load_instruments = lambda: None
    groww._download_and_load_instruments = lambda: df
    groww.get_instrument_by_groww_symbol = lambda symbol: df[df['groww_symbol'] == symbol].iloc[0].to_dict()
    return df

instruments_df = load_instruments()

# === Load Models ===
try:
    from strategy_1_model import buy_model, rr_model, compute_rsi
except Exception as e:
    st.error(f"⚠️ Failed to load models: {e}")
    st.stop()

# === Date Setup ===
start_time_ist = datetime(2025, 6, 10, 9, 15, tzinfo=ZoneInfo("Asia/Kolkata"))
end_time_ist = datetime.now(ZoneInfo("Asia/Kolkata"))

# === Pages ===
if page == "📘 Strategy Guide":
    st.title("📘 Strategy Guide")
    st.markdown("""
## 🔍 Strategy Explanation

- Combines technical indicators and machine learning models
- Uses Random Forest Classifier for Buy signal
- Uses Random Forest Regressor for Risk/Reward

### 💡 Features
- SMA, EMA, RSI, Momentum, Volatility

### 🎯 Signal Confidence
| Confidence % | Meaning          | Strength |
|--------------|------------------|----------|
| 90–100%      | Strong BUY       | 🔥        |
| 70–89%       | Moderate BUY     | ✅        |
| 50–69%       | Weak BUY         | ⚠️        |
| 30–49%       | Weak SELL        | ⚠️        |
| 10–29%       | Moderate SELL    | ❌        |
| <10%         | Strong SELL      | 💀        |

### 📊 Flow
1. Fetch data from Groww
2. Compute features
3. Predict signal & risk/reward
4. Show results live
""")
    st.stop()

if page == "📈 Live Signal":
    st.title("📈 Live Trading Signal")
    symbol = st.selectbox("Select Instrument", instruments_df['groww_symbol'].unique())
    interval_minutes = st.selectbox("Interval", [5, 10, 15, 30], index=1)

    def live_predict(symbol, interval_minutes):
        selected = groww.get_instrument_by_groww_symbol(symbol)
        data = groww.get_historical_candle_data(
            trading_symbol=selected['trading_symbol'],
            exchange=selected['exchange'],
            segment=selected['segment'],
            start_time=start_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
            end_time=end_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
            interval_in_minutes=interval_minutes
        )

        if not data.get('candles'):
            st.error("⚠️ No candle data returned.")
            return

        df = pd.DataFrame(data['candles'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Kolkata')
        df.sort_values(by='timestamp', inplace=True)

        df['SMA_10'] = df['close'].rolling(10).mean()
        df['EMA_10'] = df['close'].ewm(span=10).mean()
        df['Momentum'] = df['close'].diff(10)
        df['Volatility'] = df['close'].rolling(10).std()
        df['RSI'] = compute_rsi(df['close'])

        latest = df[['SMA_10', 'EMA_10', 'RSI', 'Momentum', 'Volatility']].dropna().tail(1)
        if latest.empty:
            st.warning("Not enough data.")
            return

        proba = buy_model.predict_proba(latest)[0]
        confidence = proba[1] * 100
        signal = "BUY" if proba[1] > 0.5 else "HOLD/SELL"
        rr_signal = rr_model.predict(latest)[0]

        # === Signal History Save ===
        log = {
            "timestamp": df['timestamp'].iloc[-1],
            "symbol": symbol,
            "signal": signal,
            "confidence": confidence,
            "rr_ratio": rr_signal
        }
        pd.DataFrame([log]).to_csv("signal_log.csv", mode='a', header=not os.path.exists("signal_log.csv"), index=False)

        # === Display ===
        st.write(f"**🕒 Time:** {log['timestamp']}")
        st.write(f"**📈 Signal:** {log['signal']}")
        st.write(f"**🎯 Confidence:** {confidence:.2f}%")
        st.write(f"**📊 Risk/Reward:** {rr_signal:.4f}")
        st.dataframe(df.tail(10), use_container_width=True)

        # Notification timer
        next_candle = df['timestamp'].iloc[-1] + timedelta(minutes=interval_minutes)
        remaining = next_candle - datetime.now(ZoneInfo("Asia/Kolkata"))
        st.info(f"⏳ Next candle in {remaining.seconds//60}m {remaining.seconds%60}s")

    live_predict(symbol, interval_minutes)

if page == "🧠 Retrain Model":
    st.title("🧠 Retrain ML Models")
    if st.button("🔁 Start Retraining"):
        with st.spinner("Retraining..."):
            try:
                process = subprocess.Popen(
                    [sys.executable, "strategy_1_retrain.py"],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    universal_newlines=True
                )
                logs = ""
                for line in process.stdout:
                    logs += line
                    st.code(logs)
                process.wait()
                if process.returncode == 0:
                    st.success("✅ Retrained successfully!")
                    st.rerun()
                else:
                    st.error("❌ Retraining failed.")
            except Exception as e:
                st.error(f"Error: {e}")

if page == "📊 Backtest":
    st.title("📊 Backtest Panel")
    st.markdown("🚧 *Work in progress — integrate historical predictions and metrics.*")

if page == "🔬 Trade Lab":
    st.title("🔬 Trade Simulation Lab")
    st.markdown("🚧 *Work in progress — simulate what-if trades based on different signals and confidence levels.*")
