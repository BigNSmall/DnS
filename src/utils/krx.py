import time
import pandas as pd
from datetime import datetime, timedelta
from pykrx import stock
from tqdm import tqdm


def get_stock_data(from_date, to_date, ticker) -> pd.DataFrame:
    df = stock.get_market_ohlcv(from_date, to_date, ticker)
    return df


def save_5y_data():
    for ticker in tqdm(stock.get_market_ticker_list()):
        today = "20240825"
        before_5y = "20190825"
        df = get_stock_data(before_5y, today, ticker)
        name = stock.get_market_ticker_name(ticker)
        df.to_parquet(f"./data/{name}_{before_5y}_{today}.parquet")
        time.sleep(1.5)


save_5y_data()
