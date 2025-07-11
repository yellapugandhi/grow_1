import pandas as pd
import datetime
from growwapi import GrowwAPI

def prepare_df(raw_data):
    df = pd.DataFrame(raw_data['candles'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    return df

def load_data(auth_token):
    groww = GrowwAPI(auth_token)
    instruments_df = pd.read_csv("instruments.csv")

    groww.instruments = instruments_df
    groww._load_instruments = lambda: None
    groww._download_and_load_instruments = lambda: instruments_df

    today = datetime.datetime.now()

    def fetch(start_offset, end_offset, interval):
        start = (today - datetime.timedelta(days=start_offset)).strftime("%Y-%m-%d 09:15:00")
        end   = (today - datetime.timedelta(days=end_offset)).strftime("%Y-%m-%d 15:15:00")
        return groww.get_historical_candle_data(
            "NSE-NIFTY", groww.EXCHANGE_NSE, groww.SEGMENT_CASH, start, end, interval
        )

    df_4 = fetch(120, 90, 240)  # ~1 candle per day
    df_3 = fetch(90, 60, 60)    # ~6 candles per day
    df_2 = fetch(60, 30, 10)    # ~30 candles per day
    df_live = fetch(30, 0, 10)  # latest 10-min candles

    return groww, df_4, df_3, df_2, df_live
