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
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


def main():
    # === Accept token ===
    token = os.getenv("GROWW_API_AUTH_TOKEN") or input("🔐 Enter your Groww API Token: ").strip()

    if not token:
        raise ValueError("❌ Groww API token not provided.")

    # === Load Data ===
    groww, df_4, df_3, _, _ = load_data(token)

    def prepare_df(raw_data):
        df = pd.DataFrame(raw_data['candles'],
                          columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        return df

    df_4 = prepare_df(df_4)
    df_3 = prepare_df(df_3)

    df = pd.concat([df_4, df_3], ignore_index=True).drop_duplicates(subset='timestamp')
    df.sort_values(by='timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # === Add Indicators ===
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['Momentum'] = df['close'] - df['close'].shift(10)
    df['Volatility'] = df['close'].rolling(window=10).std()
    df['RSI'] = compute_rsi(df['close'])
    df['BB_Upper'], df['BB_Lower'] = compute_bollinger_bands(df['close'])
    df['Stoch_K'], df['Stoch_D'] = compute_stochastic(df)
    df['MACD'], df['Signal_Line'] = compute_macd(df)

    # === Label Targets ===
    df['Buy_Signal'] = 0
    df.loc[(df['RSI'] < 30) | ((df['Momentum'] > 0) & (df['close'] > (2 * (df['SMA_10'] + df['EMA_10'])) / 3)), 'Buy_Signal'] = 1
    df.loc[~((df['RSI'] > 70) | ((df['Momentum'] < 0) & (df['close'] < (2 * (df['SMA_10'] + df['EMA_10'])) / 3))), 'Buy_Signal'] = 1
    df.loc[((df['Stoch_K'] < 20) | (df['Stoch_D'] < 20)) & (df['Buy_Signal'] == 0), 'Buy_Signal'] = 1
    df.loc[((df['MACD'] > df['Signal_Line']) & (df['Signal_Line'].isnull().all())) & (df['Buy_Signal'] == 0), 'Buy_Signal'] = 1
    df.loc[((df['close'] < df['BB_Lower']) | (df['close'] > df['BB_Upper'])) & (df['Buy_Signal'] == 0), 'Buy_Signal'] = 1

    df['Risk_Reward'] = (df['high'] - df['low']) / df['close']

    # === Clean & Balance ===
    required = ['SMA_10', 'EMA_10', 'RSI', 'Momentum', 'Volatility', 'Buy_Signal', 'Risk_Reward']
    df.dropna(subset=required, inplace=True)
    df.ffill(inplace=True)

    df_1 = df[df['Buy_Signal'] == 1]
    df_0 = df[df['Buy_Signal'] == 0].sample(n=len(df_1), replace=(len(df_1) > len(df[df['Buy_Signal'] == 0])), random_state=42)
    df_balanced = pd.concat([df_1, df_0]).sample(frac=1, random_state=42)

    # === Train Models ===
    features = ['SMA_10', 'EMA_10', 'RSI', 'Momentum', 'Volatility']
    X = df_balanced[features]
    y_buy = df_balanced['Buy_Signal']
    y_rr = df_balanced['Risk_Reward']

    X_train, X_test, y_train_buy, y_test_buy = train_test_split(X, y_buy, test_size=0.2, random_state=42)
    _, _, y_train_rr, y_test_rr = train_test_split(X, y_rr, test_size=0.2, random_state=42)

    # --- Classifier ---
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

    # --- Regressor ---
    rr_model = RandomForestRegressor(n_estimators=3000, max_depth=50, random_state=42)
    rr_model.fit(X_train, y_train_rr)

    # === Evaluate ===
    acc = accuracy_score(y_test_buy, buy_model.predict(X_test))
    mse = mean_squared_error(y_test_rr, rr_model.predict(X_test))

    print(f"✅ Buy Accuracy: {acc:.4f}")
    print(f"✅ RR MSE: {mse:.6f}")

    # === Save Models ===
    os.makedirs("models", exist_ok=True)
    joblib.dump(buy_model, "models/buy_model_latest.pkl")
    joblib.dump(rr_model, "models/rr_model_latest.pkl")
    print("💾 Models saved to models/ directory.")


if __name__ == "__main__":
    main()
