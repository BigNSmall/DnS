from utils.time_windowed_data import create_time_windows

import pandas as pd

if __name__ == "__main__":
    df = pd.read_parquet("C:/Users/yjahn/Desktop/DnS/data/NAVER_20190806_20240804.parquet")   
    print(df.head())
    td = create_time_windows(df, 12, 5)
    print(td[:2])
