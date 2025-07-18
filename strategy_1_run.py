import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from growwapi import GrowwAPI
import joblib
import numpy as np
from functools import lru_cache
import subprocess
import sys
import traceback

st.set_page_config(page_title="Trading Signal Predictor", layout="wide")

st.sidebar.title("🔐 Groww API Auth")
api_key = st.sidebar.text_input("Enter your Groww API token", type="password")
if not api_key:
    st.warning("Please enter your Groww API token in the sidebar.")
    st.stop()
groww = GrowwAPI(api_key)

page = st.sidebar.radio("📚 Navigation", ["📈 Live Signal", "🧠 Retrain Model", "📘 Strategy Guide"])

# --- Cached Instrument Metadata
@lru_cache(maxsize=1)
def load_instruments():
    try:
        df = pd.read_csv("instruments.csv")
        groww.instruments = df
        groww._load_instruments = lambda: None
        groww._download_and_load_instruments = lambda: df
        groww.get_instrument_by_groww_symbol = lambda symbol: df[df['groww_symbol'] == symbol].iloc[0].to_dict()
        return df
    except Exception as e:
        st.error(f"Failed to load instruments.csv: {e}")
        st.stop()

instruments_df = load_instruments()

# --- Symbol Picker
symbols_list = instruments_df["groww_symbol"].sort_values().unique().tolist()
default_symbol = "NSE-NIFTY" if "NSE-NIFTY" in symbols_list else symbols_list[0]
selected_symbol = st.sidebar.selectbox("Select Symbol", symbols_list, index=symbols_list.index(default_symbol))

try:
    from strategy_1_model import buy_model, rr_model, compute_rsi, compute_macd
except Exception as e:
    st.error(f"⚠️ Failed to load models: {e}\n{traceback.format_exc()}")
    st.stop()

now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
if datetime.strptime("09:15", "%H:%M").time() <= now_ist.time() <= datetime.strptime("15:30", "%H:%M").time():
    st.markdown("<meta http-equiv='refresh' content='600'>", unsafe_allow_html=True)  # Every 10 mins

strategy_option = st.sidebar.selectbox("Select Strategy Version", ["Strategy 1"])

if page == "📘 Strategy Guide":
    st.title("📘 Strategy Guide – Signal Confidence Strength")
    st.markdown("""
### 🔍 How Confidence Works:
- The model outputs a probability that a candle is a **BUY signal**.
- We translate this into **labels** for human-friendly interpretation.

### 🧠 Confidence Strength Breakdown

| Confidence %       | Label              | Meaning                                  |
|--------------------|--------------------|------------------------------------------|
| **90–100%**        | 🔥 Strong BUY      | Very high conviction — ideal entry zone. |
| **70–89.99%**      | ✅ Moderate BUY    | Good signal, with decent model backing.  |
| **50–69.99%**      | ⚠️ Weak BUY        | Slight positive bias — wait for confirmation. |
| **30–49.99%**      | ⚠️ Weak SELL       | Slight negative bias — stay cautious.    |
| **10–29.99%**      | ❌ Moderate SELL   | Selling pressure likely — avoid buying.  |
| **0–9.99%**        | 💀 Strong SELL     | Very bearish — exit or short if applicable. |

---
- Combine this with technical indicators (RSI, MACD) for better decisions.
- Use **Strong BUY** or **Strong SELL** as clear entry/exit zones.
- For **Weak signals**, observe next candles or add filters.
    """)
    st.stop()

# --- Feature Engineering for Live Prediction
def streamlit_features(df):
    df['SMA_10'] = df['close'].rolling(10).mean()
    df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['Momentum'] = df['close'] - df['close'].shift(5)
    df['Volatility'] = df['close'].rolling(10).std()
    df['RSI'] = compute_rsi(df['close'])
    df['Lag_Close'] = df['close'].shift(1)
    df['Lag_Momentum'] = df['Momentum'].shift(1)
    df['MACD'], df['MACD_Signal'] = compute_macd(df['close'])
    # Must match order used at train time:
    final_cols = ["SMA_10", "EMA_10", "RSI", "Momentum", "Volatility",
                  "Lag_Close", "Lag_Momentum", "MACD", "MACD_Signal"]
    return df[final_cols]

if page == "📈 Live Signal":
    def live_predict(symbol=selected_symbol, interval_minutes=10):
        safe_duration_minutes = interval_minutes * 1400
        end_time = datetime.now(ZoneInfo("Asia/Kolkata"))
        start_time = end_time - timedelta(minutes=safe_duration_minutes)
        selected = groww.get_instrument_by_groww_symbol(symbol)

        try:
            data = groww.get_historical_candle_data(
                trading_symbol=selected['trading_symbol'],
                exchange=selected['exchange'],
                segment=selected['segment'],
                start_time=start_time.strftime("%Y-%m-%d %H:%M:%S"),
                end_time=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                interval_in_minutes=interval_minutes
            )
        except Exception as e:
            st.error(f"Groww API error: {e}")
            return

        if isinstance(data, dict) and 'candles' in data and len(data['candles']) > 0:
            df = pd.DataFrame(data['candles'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Kolkata')
            df.sort_values(by='timestamp', inplace=True)
            df.reset_index(drop=True, inplace=True)
            df.fillna(method='ffill', inplace=True)

            features_df = streamlit_features(df).dropna().tail(1)
            if features_df.empty:
                st.warning("Not enough data to predict.")
                return

            proba = buy_model.predict_proba(features_df)[0]
            confidence = proba[1] * 100
            buy_signal = int(proba[1] > 0.5)
            rr_signal = rr_model.predict(features_df)[0]

            st.markdown("### 🕒 Last Candle")
            st.markdown(f"**{df['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S')}**")

            st.markdown("### 📈 Signal")
            st.markdown(f"**{'BUY' if buy_signal == 1 else 'HOLD / SELL'}**")

            if confidence >= 90:
                strength = "🔥 Strong BUY"
                color = "green"
            elif confidence >= 70:
                strength = "✅ Moderate BUY"
                color = "green"
            elif confidence >= 50:
                strength = "⚠️ Weak BUY"
                color = "orange"
            elif confidence >= 30:
                strength = "⚠️ Weak SELL"
                color = "orange"
            elif confidence >= 10:
                strength = "❌ Moderate SELL"
                color = "red"
            else:
                strength = "💀 Strong SELL"
                color = "darkred"

            st.markdown(
                f"### 🎯 <b>Confidence:</b> <span style='color:{color}'>{confidence:.2f}% - {strength}</span>",
                unsafe_allow_html=True
            )
            st.markdown(f"### 📊 <b>Risk/Reward:</b> `{rr_signal:.4f}`", unsafe_allow_html=True)
            st.dataframe(df.tail(10), use_container_width=True)
        else:
            st.error("⚠️ No candle data returned from Groww API.")

    live_predict()

    if st.button("🔃 Refresh Now"):
        st.rerun()

if page == "🧠 Retrain Model":
    st.title("🧠 Retrain Trading Models")
    if st.button("🔁 Start Retraining"):
        st.info("📱 Starting retraining...")
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
                st.success("✅ Retraining complete!")
                st.rerun()
            else:
                st.error("❌ Retraining failed. Please check strategy_1_retrain.py.")
        except Exception as e:
            st.error(f"❌ Error during retraining: {e}")
