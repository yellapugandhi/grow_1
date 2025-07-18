import os
import pandas as pd
import datetime
import pytz
from dotenv import load_dotenv
from growwapi import GrowwAPI
import warnings
import time
warnings.filterwarnings("ignore")

load_dotenv()
AUTH_TOKEN = os.getenv("GROWW_AUTH_TOKEN")

def prepare_df(raw_data):
    df = pd.DataFrame(raw_data['candles'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
    return df

def initialize_groww_api(auth_token, instruments_path="instruments.csv"):
    groww = GrowwAPI(auth_token)
    instruments_df = pd.read_csv(instruments_path, low_memory=False)
    groww.instruments = instruments_df
    groww._load_instruments = lambda: None
    groww._download_and_load_instruments = lambda: instruments_df
    return groww

def fetch_candles_chunk(groww, symbol, start, end, interval):
    max_daily = {
        1: 7,       # 1 min - 7 days
        5: 15,      # 5 min - 15 days
        10: 30,     # 10 min - 30 days
        15: 31,     # 15 min - 31 days
        30: 90,     # 30 min - 90 days
        60: 150,    # 60 min - 150 days
        240: 365,   # 4h - 365 days
        1440: 1080, # 1d - 1080 days (~3y)
    }
    days = (end - start).days
    if interval not in max_daily or days > max_daily[interval]:
        print(f"Skipping: {interval}m interval not allowed for {days} days")
        return None
    try:
        print(f"Fetching {symbol}: {start} -> {end} ({interval}m)")
        data = groww.get_historical_candle_data(
            trading_symbol=symbol,
            exchange=groww.EXCHANGE_NSE,
            segment=groww.SEGMENT_CASH,
            start_time=start.strftime("%Y-%m-%d %H:%M:%S"),
            end_time=end.strftime("%Y-%m-%d %H:%M:%S"),
            interval_in_minutes=interval
        )
        time.sleep(2)  # avoid rate limiting
        return data if data and data.get("candles") else None
    except Exception as e:
        print(f"API failed: {e}")
        return None

def load_data(symbol="NIFTY"):
    groww = initialize_groww_api(AUTH_TOKEN)
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.datetime.now(tz=ist).replace(hour=15, minute=15, second=0, microsecond=0)

    # Each tuple: (label, days_ago_start, days_ago_end, interval)
    periods = [
        ("df_live", 0, 30, 10),
        ("df_1", 30, 60, 15),
        ("df_2", 60, 90, 30),
        ("df_3", 90, 180, 1440),     # Use daily candle for older data
        ("df_4", 180, 360, 1440),    # Use daily candle for even older data
    ]

    dfs = {}
    for label, ago_start, ago_end, interval in periods:
        start = now - datetime.timedelta(days=ago_end)
        end = now - datetime.timedelta(days=ago_start)
        chunk = fetch_candles_chunk(groww, symbol, start, end, interval)
        if chunk is not None:
            dfs[label] = prepare_df(chunk)
        else:
            print(f"Skipping {label} - could not fetch data (interval={interval}, {start}→{end})")
    return groww, dfs

if __name__ == "__main__":
    print("Ready to Groww!")
    groww, dfs = load_data()
    for label, df in dfs.items():
        print(f"\n📊 {label} - {len(df)} rows | Columns: {df.columns.tolist()}")
        print(df.head(5))
    if dfs:
        df_master = pd.concat(dfs.values(), ignore_index=True).sort_values(by="timestamp").drop_duplicates()
        print(f"\nMaster DataFrame with {len(df_master)} rows, {len(dfs)} segments.")
