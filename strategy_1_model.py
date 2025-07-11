import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
import warnings

from data import load_data

warnings.filterwarnings("ignore")

buy_model = None
rr_model = None

# === Load Token from Environment or Streamlit Injection ===
AUTH_TOKEN = os.getenv("GROWW_API_AUTH_TOKEN")

# Fallback: check if token is injected (from Streamlit)
if not AUTH_TOKEN and "__streamlit_groww_token__" in globals():
    AUTH_TOKEN = __streamlit_groww_token__

# Fallback: prompt for token (only if running from CLI)
if not AUTH_TOKEN:
    try:
        AUTH_TOKEN = input("🔐 Enter your Groww API Token: ").strip()
    except EOFError:
        raise EnvironmentError("❌ GROWW_API_AUTH_TOKEN not set and no input provided.")

if not AUTH_TOKEN:
    raise EnvironmentError("⚠️ GROWW_API_AUTH_TOKEN not set in environment, Streamlit, or prompt.")

# === Load Data (first-time only df_4 and df_3) ===
groww, df_4, df_3, _, _ = load_data(AUTH_TOKEN)

def prepare_df(raw_data):
    df = pd.DataFrame(raw_data['candles'],
                      columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    return df

df_4 = prepare_df(df_4)
df_3 = prepare_df(df_3)

# === Combine Data ===
df = pd.concat([df_4, df_3], ignore_index=True)
df.drop_duplicates(subset=['timestamp'], inplace=True)
df.sort_values(by='timestamp', inplace=True)
df.reset_index(drop=True, inplace=True)

# === Feature Engineering ===
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_bollinger_bands(series, period=20):
    ma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    return ma + 2 * std, ma - 2 * std

def compute_stochastic(df):
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    k = ((df['close'] - low_14) / (high_14 - low_14)) * 100
    d = k.rolling(window=3).mean()
    return k, d

def compute_macd(df):
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

# === Indicators ===
df['SMA_10'] = df['close'].rolling(window=10).mean()
df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
df['Momentum'] = df['close'] - df['close'].shift(10)
df['Volatility'] = df['close'].rolling(window=10).std()
df['RSI'] = compute_rsi(df['close'])
bb_u, bb_l = compute_bollinger_bands(df['close'])
df['BB_Upper'] = bb_u
df['BB_Lower'] = bb_l
k, d = compute_stochastic(df)
df['Stoch_K'] = k
df['Stoch_D'] = d
macd, sig_line = compute_macd(df)
df['MACD'] = macd
df['Signal_Line'] = sig_line

# === Label Generation ===
df['Buy_Signal'] = 0
df.loc[(df['RSI'] < 30) | ((df['Momentum'] > 0) & (df['close'] > (2 * (df['SMA_10'] + df['EMA_10'])) / 3)), 'Buy_Signal'] = 1
df.loc[~((df['RSI'] > 70) | ((df['Momentum'] < 0) & (df['close'] < (2 * (df['SMA_10'] + df['EMA_10'])) / 3))), 'Buy_Signal'] = 1
df.loc[((k < 20) | (d < 20)) & (df['Buy_Signal'] == 0), 'Buy_Signal'] = 1
df.loc[((macd > sig_line) & (sig_line.isnull().all())) & (df['Buy_Signal'] == 0), 'Buy_Signal'] = 1
df.loc[((df['close'] < df['BB_Lower']) | (df['close'] > df['BB_Upper'])) & (df['Buy_Signal'] == 0), 'Buy_Signal'] = 1

df['Risk_Reward'] = (df['high'] - df['low']) / df['close']

# === Clean + Balance Dataset ===
required = ['SMA_10', 'EMA_10', 'RSI', 'Momentum', 'Volatility', 'Buy_Signal', 'Risk_Reward']
df.dropna(subset=required, inplace=True)
df.ffill(inplace=True)

# Balance positive/negative
df_pos = df[df['Buy_Signal'] == 1]
n_pos = len(df_pos)
df_neg = df[df['Buy_Signal'] == 0]
n_neg = len(df_neg)
df_neg_sample = df_neg.sample(n=n_pos, replace=(n_pos > n_neg), random_state=42)
df_balanced = pd.concat([df_pos, df_neg_sample]).sample(frac=1, random_state=42)

# === Training Function for Streamlit ===
def train_and_save_models():
    global buy_model, rr_model

    features = ['SMA_10', 'EMA_10', 'RSI', 'Momentum', 'Volatility']
    X = df_balanced[features]
    y_buy = df_balanced['Buy_Signal']
    y_rr = df_balanced['Risk_Reward']

    X_train, X_test, y_train_buy, y_test_buy = train_test_split(X, y_buy, test_size=0.2, random_state=42)
    _, _, y_train_rr, y_test_rr = train_test_split(X, y_rr, test_size=0.2, random_state=42)

    clf_params = {
        'n_estimators': [300, 500, 1000],
        'max_depth': [10, 30, 50, None],
        'min_samples_split': [2, 5, 10]
    }
    clf_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        clf_params,
        n_iter=5,
        scoring='accuracy',
        cv=3,
        random_state=42
    )
    clf_search.fit(X_train, y_train_buy)
    buy_model = clf_search.best_estimator_

    rr_model = RandomForestRegressor(n_estimators=3000, max_depth=50, random_state=42)
    rr_model.fit(X_train, y_train_rr)

    print("✅ Buy Accuracy:", accuracy_score(y_test_buy, buy_model.predict(X_test)))
    print("✅ RR MSE:", mean_squared_error(y_test_rr, rr_model.predict(X_test)))

    os.makedirs("models", exist_ok=True)
    joblib.dump(buy_model, "models/buy_model_latest.pkl")
    joblib.dump(rr_model, "models/rr_model_latest.pkl")
    print("💾 Models saved.")

# === Load models for use ===
buy_model = joblib.load("models/buy_model_latest.pkl")
rr_model = joblib.load("models/rr_model_latest.pkl")

# === RSI export ===
def compute_rsi_for_live(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

compute_rsi = compute_rsi_for_live
