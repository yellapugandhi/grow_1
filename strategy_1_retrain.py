import os
import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from data import load_data

warnings.filterwarnings("ignore")

# === Load Candle Data ===
groww, data_frames = load_data()
# Accept whatever segments are available (robust)
ordered_labels = ["df_4", "df_3", "df_2", "df_1", "df_live"]
dfs_available = [data_frames[lbl] for lbl in ordered_labels if lbl in data_frames]
if not dfs_available:
    raise ValueError("No data segments found for training.")
df = pd.concat(dfs_available, ignore_index=True)

df.drop_duplicates(subset=["timestamp"], inplace=True)
df.sort_values(by="timestamp", inplace=True)
df.reset_index(drop=True, inplace=True)

# === Feature Engineering ===
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))

def compute_macd(series, span1=12, span2=26, span_signal=9):
    ema1 = series.ewm(span=span1, adjust=False).mean()
    ema2 = series.ewm(span=span2, adjust=False).mean()
    macd = ema1 - ema2
    signal = macd.ewm(span=span_signal, adjust=False).mean()
    return macd, signal

df["SMA_10"] = df["close"].rolling(10).mean()
df["EMA_10"] = df["close"].ewm(span=10, adjust=False).mean()
df["Momentum"] = df["close"] - df["close"].shift(5)
df["Volatility"] = df["close"].rolling(10).std()
df["RSI"] = compute_rsi(df["close"])
df["Lag_Close"] = df["close"].shift(1)
df["Lag_Momentum"] = df["Momentum"].shift(1)
df["MACD"], df["MACD_Signal"] = compute_macd(df["close"])

df["Buy_Signal"] = 0
df.loc[(df["RSI"] < 45) & (df["Momentum"] > 0), "Buy_Signal"] = 1
df["Risk_Reward"] = (df["high"] - df["low"]) / df["close"]

required_cols = ["SMA_10", "EMA_10", "RSI", "Momentum", "Volatility", "Lag_Close", "Lag_Momentum", "MACD", "MACD_Signal", "Buy_Signal", "Risk_Reward"]
df.dropna(subset=required_cols, inplace=True)
df.ffill(inplace=True)

print("CLASS BALANCE:")
print(df['Buy_Signal'].value_counts())

features = ["SMA_10", "EMA_10", "RSI", "Momentum", "Volatility", "Lag_Close", "Lag_Momentum", "MACD", "MACD_Signal"]
X = df[features]
y_buy = df["Buy_Signal"]
y_rr = df["Risk_Reward"]

# === Chronological Train/Test Split (out-of-sample, last 20% as test set) ===
split_idx = int(0.8 * len(X))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train_buy, y_test_buy = y_buy.iloc[:split_idx], y_buy.iloc[split_idx:]
y_train_rr, y_test_rr = y_rr.iloc[:split_idx], y_rr.iloc[split_idx:]

print(f"TRAIN size: {len(X_train)}, TEST size: {len(X_test)}")
    
# === Regularized Classifier Training ===
forest_params = {
    "n_estimators": [100, 200],
    "max_depth": [2, 3, 5],
    "min_samples_leaf": [10, 20, 30],
}

clf_search = GridSearchCV(
    RandomForestClassifier(class_weight="balanced", random_state=42),
    forest_params,
    scoring="roc_auc",
    cv=TimeSeriesSplit(n_splits=4),
    n_jobs=-1
)
clf_search.fit(X_train, y_train_buy)
buy_model = clf_search.best_estimator_

# === Regressor Training ===
rr_model = RandomForestRegressor(n_estimators=100, max_depth=3, min_samples_leaf=15, random_state=42)
rr_model.fit(X_train, y_train_rr)

# === Evaluation ===
y_pred_buy = buy_model.predict(X_test)
y_pred_prob = buy_model.predict_proba(X_test)[:, 1]
y_pred_rr = rr_model.predict(X_test)
acc = accuracy_score(y_test_buy, y_pred_buy)
mse = mean_squared_error(y_test_rr, y_pred_rr)
roc_auc = roc_auc_score(y_test_buy, y_pred_prob)
print(f"\n🎯 Model Evaluation [Out-of-sample]:")
print(f"✅ Buy Accuracy     : {acc:.4f}")
print(f"✅ ROC AUC          : {roc_auc:.4f}")
print(f"✅ Risk/Reward MSE  : {mse:.4f}")

print("\n📋 Classification Report (Buy Signal):")
print(classification_report(y_test_buy, y_pred_buy, digits=4))
print("\nConfusion matrix:")
print(confusion_matrix(y_test_buy, y_pred_buy))

# === Save Models ===
os.makedirs("models", exist_ok=True)
joblib.dump(buy_model, "models/buy_model_latest.pkl")
joblib.dump(rr_model, "models/rr_model_latest.pkl")
print("\n💾 Models saved to 'models/' directory.")
