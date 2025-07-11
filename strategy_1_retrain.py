import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
import warnings

from data import load_data

warnings.filterwarnings("ignore")

# === Auth Token from ENV ===
AUTH_TOKEN = os.getenv("GROWW_API_AUTH_TOKEN")
if not AUTH_TOKEN:
    AUTH_TOKEN = input("🔐 Enter your Groww API Token: ").strip()

# === Load Data ===
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

df['SMA_10'] = df['close'].rolling(window=10).mean()
df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
df['Momentum'] = df['close'] - df['close'].shift(10)
df['Volatility'] = df['close'].rolling(window=10).std()
df['RSI'] = compute_rsi(df['close'])

df['Buy_Signal'] = 0
df.loc[(df['RSI'] < 30) & (df['Momentum'] > 0), 'Buy_Signal'] = 1

df['Risk_Reward'] = (df['high'] - df['low']) / df['close']

# === Clean Data ===
required = ['SMA_10', 'EMA_10', 'RSI', 'Momentum', 'Volatility', 'Buy_Signal', 'Risk_Reward']
df.dropna(subset=required, inplace=True)
df.ffill(inplace=True)

# === Balance Data ===
df_pos = df[df['Buy_Signal'] == 1]
df_neg = df[df['Buy_Signal'] == 0].sample(n=len(df_pos), random_state=42)
df_balanced = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=42)

# === Train Models ===
features = ['SMA_10', 'EMA_10', 'RSI', 'Momentum', 'Volatility']
X = df_balanced[features]
y_buy = df_balanced['Buy_Signal']
y_rr = df_balanced['Risk_Reward']

X_train, X_test, y_train_buy, y_test_buy = train_test_split(X, y_buy, test_size=0.2, random_state=42)
_, _, y_train_rr, y_test_rr = train_test_split(X, y_rr, test_size=0.2, random_state=42)

clf_params = {
    'n_estimators': [100, 300, 500],
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

rr_model = RandomForestRegressor(n_estimators=500, max_depth=50, random_state=42)
rr_model.fit(X_train, y_train_rr)

# === Evaluation ===
print("✅ Buy Accuracy:", accuracy_score(y_test_buy, buy_model.predict(X_test)))
print("✅ RR MSE:", mean_squared_error(y_test_rr, rr_model.predict(X_test)))

# === Save Models ===
os.makedirs("models", exist_ok=True)
joblib.dump(buy_model, "models/buy_model_latest.pkl")
joblib.dump(rr_model, "models/rr_model_latest.pkl")
print("💾 Models saved in 'models/' directory.")
