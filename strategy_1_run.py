import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from growwapi import GrowwAPI
import joblib
from functools import lru_cache
import subprocess
import sys
import os

st.set_page_config(page_title="Trading Signal Predictor", layout="wide")

# === Groww API Auth ===
st.sidebar.title("\U0001F510 Groww API Auth")
api_key = st.sidebar.text_input("Enter your Groww API token", type="password")

if not api_key:
    st.warning("Please enter your Groww API token in the sidebar.")
    st.stop()

groww = GrowwAPI(api_key)

# === Navigation ===
page = st.sidebar.radio("\U0001F4DA Navigation", ["\U0001F4C8 Live Signal", "\U0001F9E0 Retrain Model", "\U0001F4D8 Strategy Guide"])

# === Load instrument metadata ===
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
    st.error(f"\u26A0\ufe0f Failed to load models: {e}")
    st.stop()

# === Strategy Dropdown ===
strategy_option = st.sidebar.selectbox("Select Strategy Version", ["Strategy 1"])
symbol_option = st.sidebar.selectbox("Select Instrument", instruments_df['groww_symbol'].unique(), index=0)

# === Trading hours auto-refresh ===
now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
if datetime.strptime("09:15", "%H:%M").time() <= now_ist.time() <= datetime.strptime("15:30", "%H:%M").time():
    st.markdown("<meta http-equiv='refresh' content='600'>", unsafe_allow_html=True)

# === Guide Page ===
if page == "\U0001F4D8 Strategy Guide":
    st.title("\U0001F4D8 Strategy Guide")
    st.markdown("""
### 🔍 Strategy Explanation

This model uses:
- SMA, EMA, RSI, Momentum, Volatility
- Random Forests for signal + risk/reward

#### 🟢 Signal Strength Table:

| Confidence % | Meaning         | Strength |
|--------------|------------------|----------|
| 90%+         | Strong BUY       | 🔥 |
| 70–89%       | Moderate BUY     | ✅ |
| 50–69%       | Weak BUY         | ⚠ |
| 30–49%       | Weak SELL        | ⚠ |
| 10–29%       | Moderate SELL    | ❌ |
| <10%         | Strong SELL      | 💀 |

#### ⚙ Strategy Flow:
1. Fetch data
2. Feature Engineering
3. Predict Signal & Risk/Reward
4. Show Live Output
""")
    st.stop()

# === Live Signal ===
if page == "\U0001F4C8 Live Signal":
    def live_predict(symbol=symbol_option, interval_minutes=10):
        start_time_ist = datetime(2025, 6, 10, 9, 15, tzinfo=ZoneInfo("Asia/Kolkata"))
        end_time_ist = datetime.now(ZoneInfo("Asia/Kolkata"))

        selected = groww.get_instrument_by_groww_symbol(symbol)

        data = groww.get_historical_candle_data(
            trading_symbol=selected['trading_symbol'],
            exchange=selected['exchange'],
            segment=selected['segment'],
            start_time=start_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
            end_time=end_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
            interval_in_minutes=interval_minutes
        )

        if isinstance(data, dict) and 'candles' in data and len(data['candles']) > 0:
            df = pd.DataFrame(data['candles'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Kolkata')
            df.sort_values(by='timestamp', inplace=True)
            df.reset_index(drop=True, inplace=True)

            df['SMA_10'] = df['close'].rolling(window=10).mean()
            df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
            df['Momentum'] = df['close'] - df['close'].shift(10)
            df['Volatility'] = df['close'].rolling(window=10).std()
            df['RSI'] = compute_rsi(df['close'])

            latest = df[['SMA_10', 'EMA_10', 'RSI', 'Momentum', 'Volatility']].dropna().tail(1)

            if latest.empty:
                st.warning("Not enough data to predict.")
                return

            proba = buy_model.predict_proba(latest)[0]
            confidence = proba[1] * 100
            buy_signal = int(proba[1] > 0.5)
            rr_signal = rr_model.predict(latest)[0]

            st.markdown(f"### \u{1F550} Last Candle: **{df['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S')}**")
            st.markdown(f"### \u{1F4C8} Signal: **{'BUY' if buy_signal == 1 else 'HOLD / SELL'}**")

            if confidence >= 90:
                strength = "🔥 Strong BUY"
                color = "green"
            elif confidence >= 70:
                strength = "✅ Moderate BUY"
                color = "green"
            elif confidence >= 50:
                strength = "⚠ Weak BUY"
                color = "orange"
            elif confidence >= 30:
                strength = "⚠ Weak SELL"
                color = "orange"
            elif confidence >= 10:
                strength = "❌ Moderate SELL"
                color = "red"
            else:
                strength = "💀 Strong SELL"
                color = "darkred"

            st.markdown(f"### 🎯 <b>Confidence:</b> <span style='color:{color}'>{confidence:.2f}% - {strength}</span>", unsafe_allow_html=True)
            st.markdown(f"### 📊 <b>Risk/Reward:</b> `{rr_signal:.4f}`", unsafe_allow_html=True)
            st.dataframe(df.tail(10), use_container_width=True)

            next_candle_time = df['timestamp'].iloc[-1] + timedelta(minutes=interval_minutes)
            remaining = next_candle_time - datetime.now(ZoneInfo("Asia/Kolkata"))
            st.info(f"⏳ Time until next candle: {remaining.seconds // 60}m {remaining.seconds % 60}s")
        else:
            st.error("\u26A0\ufe0f No candle data returned from Groww API.")

    live_predict()

    if st.button("\U0001F503 Refresh Now"):
        st.rerun()

# === Retrain ===
if page == "\U0001F9E0 Retrain Model":
    st.title("\U0001F9E0 Retrain Trading Models")
    if st.button("\U0001F501 Start Retraining"):
        st.info("\U0001F4E1 Starting retraining...")
        try:
            process = subprocess.Popen(
                [sys.executable, "strategy_1_retrain.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            output_area = st.empty()
            logs = ""
            for line in process.stdout:
                logs += line
                output_area.code(logs)

            process.wait()
            if process.returncode == 0:
                st.success("\u2705 Retraining complete!")
                st.rerun()
            else:
                st.error("\u274C Retraining failed. Please check strategy_1_retrain.py.")
        except Exception as e:
            st.error(f"\u274C Error during retraining: {e}")
