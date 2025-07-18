import os
import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from data import load_data

warnings.filterwarnings("ignore")

# === Load Data ===
groww, data_frames = load_data()
required_keys = {"df_2", "df_3", "df_4", "df_live"}
missing_keys = required_keys - set(data_frames.keys())
if missing_keys:
    raise ValueError(f"❌ Missing required DataFrames: {missing_keys}")

df = pd.concat([
    data_frames["df_4"],
    data_frames["df_3"],
    data_frames["df_2"],
    data_frames["df_live"]
], ignore_index=True)

df.drop_duplicates(subset=["timestamp"], inplace=True)
df.sort_values(by="timestamp", inplace=True)
df.reset_index(drop=True, inplace=True)

# === Feature Engineering ===
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df["SMA_10"] = df["close"].rolling(10).mean()
df["EMA_10"] = df["close"].ewm(span=10, adjust=False).mean()
df["Momentum"] = df["close"] - df["close"].shift(5)
df["Volatility"] = df["close"].rolling(10).std()
df["RSI"] = compute_rsi(df["close"])

# === Outcome-Based Labeling ===
forward_shift = 3
profit_threshold = 0.01  # 1% rise

df["Forward_Close"] = df["close"].shift(-forward_shift)
df["Buy_Signal"] = (df["Forward_Close"] > df["close"] * (1 + profit_threshold)).astype(int)
df["Risk_Reward"] = (df["high"] - df["low"]) / df["close"]

# === Clean & Validate ===
required_cols = ["SMA_10", "EMA_10", "RSI", "Momentum", "Volatility", "Buy_Signal", "Risk_Reward"]
df.dropna(subset=required_cols, inplace=True)
df.ffill(inplace=True)

# === Signal Stats ===
df_pos = df[df["Buy_Signal"] == 1]
df_neg = df[df["Buy_Signal"] == 0]
print(f"\n📊 Buy signals: {len(df_pos)} | Hold signals: {len(df_neg)}")

if len(df_pos) < 10:
    raise ValueError("❌ Too few Buy signals — try reducing profit_threshold or shift horizon.")

# === Balance Dataset ===
df_neg_sampled = df_neg.sample(n=len(df_pos), random_state=42)
df_balanced = pd.concat([df_pos, df_neg_sampled]).sample(frac=1, random_state=42)

if df_balanced.empty:
    raise ValueError("❌ Balanced dataset is empty — training aborted.")

# === Split Data ===
features = ["SMA_10", "EMA_10", "RSI", "Momentum", "Volatility"]
X = df_balanced[features]
y_buy = df_balanced["Buy_Signal"]
y_rr = df_balanced["Risk_Reward"]

X_train, X_test, y_train_buy, y_test_buy = train_test_split(X, y_buy, test_size=0.2, random_state=42)
_, _, y_train_rr, y_test_rr = train_test_split(X, y_rr, test_size=0.2, random_state=42)

# === Buy Classifier Training ===
clf_params = {
    "n_estimators": [100, 300],
    "max_depth": [5, 10],
    "min_samples_split": [5, 10],
    "min_samples_leaf": [3]
}

clf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    clf_params,
    n_iter=4,
    scoring="accuracy",
    cv=min(3, len(df_pos)),
    random_state=42
)
clf_search.fit(X_train, y_train_buy)
buy_model = clf_search.best_estimator_

# === Risk/Reward Regressor Training ===
rr_model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
rr_model.fit(X_train, y_train_rr)

# === Evaluation ===
acc = accuracy_score(y_test_buy, buy_model.predict(X_test))
mse = mean_squared_error(y_test_rr, rr_model.predict(X_test))

print(f"\n🎯 Model Evaluation:")
print(f"✅ Buy Accuracy     : {acc:.4f}")
print(f"✅ Risk/Reward MSE  : {mse:.4f}")
print("\n📋 Classification Report (Buy Signal):")
print(classification_report(y_test_buy, buy_model.predict(X_test), digits=4))

# === Save Models ===
os.makedirs("models", exist_ok=True)
joblib.dump(buy_model, "models/buy_model_latest.pkl")
joblib.dump(rr_model, "models/rr_model_latest.pkl")
print("\n💾 Models saved to 'models/' directory.")
