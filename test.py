import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
import warnings
warnings.filterwarnings("ignore")

# ---------- Load First-Time Data (ONLY df_4 and df_3) ----------
from data import df_4, df_3  # Do NOT import df2 or new_live_data here

# Convert raw format to DataFrame
def prepare_df(raw_data):
    df = pd.DataFrame(raw_data['candles'],
                      columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    return df

df_4 = prepare_df(df_4)
df_3 = prepare_df(df_3)

# Combine and clean
df = pd.concat([df_4, df_3], ignore_index=True)
df.drop_duplicates(subset=['timestamp'], inplace=True)
df.sort_values(by='timestamp', inplace=True)
df.reset_index(drop=True, inplace=True)

# ---------- Feature Engineering ----------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['SMA_10'] = df['close'].rolling(window=10).mean()
df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
df['Momentum'] = df['close'] - df['close'].shift(10)
df['Volatility'] = df['close'].rolling(window=10).std()
df['RSI'] = compute_rsi(df['close'])

# Targets
df['Buy_Signal'] = 0
df.loc[(df['RSI'] < 30) | ((df['Momentum'] > 0) & (df['close'] > (2 * (df['SMA_10'] + df['EMA_10'])) / 3)), 'Buy_Signal'] = 1
df.loc[~((df['RSI'] > 70) | ((df['Momentum'] < 0) & (df['close'] < (2 * (df['SMA_10'] + df['EMA_10'])) / 3))), 'Buy_Signal'] = 1
df['Risk_Reward'] = (df['high'] - df['low']) / df['close']

# Drop rows with NaNs
required_columns = ['SMA_10', 'EMA_10', 'RSI', 'Momentum', 'Volatility', 'Buy_Signal', 'Risk_Reward']
df.dropna(subset=required_columns, inplace=True)
df.ffill(inplace=True)

# ---------- Balance Buy Signal ----------
# Count positive class instances
df_1 = df[df['Buy_Signal'] == 1]
n_pos = len(df_1)
# Count negative class instances
n_neg = len(df[df['Buy_Signal'] == 0])

# Sample negative class
if n_neg >= n_pos:
    df_0 = df[df['Buy_Signal'] == 0].sample(n=n_pos, random_state=42)
else:
    df_0 = df[df['Buy_Signal'] == 0].sample(n=n_pos, replace=True, random_state=42)

# Combine and shuffle
df_balanced = pd.concat([df_1, df_0]).sample(frac=1, random_state=42)

# ---------- Train Models ----------
features = ['SMA_10', 'EMA_10', 'RSI', 'Momentum', 'Volatility']
X = df_balanced[features]
y_buy = df_balanced['Buy_Signal']
y_rr = df_balanced['Risk_Reward']

# Split for both models
X_train, X_test, y_train_buy, y_test_buy = train_test_split(X, y_buy, test_size=0.2, random_state=42)
_, _, y_train_rr, y_test_rr = train_test_split(X, y_rr, test_size=0.2, random_state=42)

# Hyperparameter tuning for classifier
clf_params = {
    'n_estimators': [ 300, 500, 1000],
    'max_depth': [10, 30, 50, None],
    'min_samples_split': [2, 5, 10]
}
clf_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), clf_params,
                                n_iter=5, scoring='accuracy', cv=3, random_state=42)
clf_search.fit(X_train, y_train_buy)
buy_model = clf_search.best_estimator_

# Train regressor
rr_model = RandomForestRegressor(n_estimators=3000, max_depth=50, random_state=42)
rr_model.fit(X_train, y_train_rr)

# ---------- Evaluate ----------
y_pred_buy = buy_model.predict(X_test)
y_pred_rr = rr_model.predict(X_test)

print("✅ Buy Model Accuracy:", accuracy_score(y_test_buy, y_pred_buy))
print("✅ Risk-Reward MSE:", mean_squared_error(y_test_rr, y_pred_rr))

# ---------- Save Models ----------
os.makedirs("models", exist_ok=True)
joblib.dump(buy_model, 'models/buy_model_latest.pkl')
joblib.dump(rr_model, 'models/rr_model_latest.pkl')

print("💾 First-time models saved in 'models/' directory.")