import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
import matplotlib.pyplot as plt


def calculate_rsi(price_series: pd.Series, window: int = 14) -> pd.Series:
    """
    주어진 가격 시리즈에 대한 RSI(Relative Strength Index)를 계산합니다.

    :param price_series: 가격 데이터를 포함하는 pandas Series
    :param window: RSI 계산을 위한 기간 (기본값: 14)
    :return: 계산된 RSI 값을 포함하는 pandas Series
    """
    # RSIIndicator 객체 생성
    rsi_indicator = RSIIndicator(close=price_series, window=window)

    # RSI 계산
    rsi = rsi_indicator.rsi()

    return rsi


def visualize_price_and_rsi(price_series: pd.Series, rsi_series: pd.Series):
    """
    가격 데이터와 RSI 지표를 시각화합니다.

    :param price_series: 가격 데이터를 포함하는 pandas Series
    :param rsi_series: RSI 데이터를 포함하는 pandas Series
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # 가격 그래프
    ax1.plot(price_series.index, price_series.values)
    ax1.set_title("Price")
    ax1.set_ylabel("Price")

    # RSI 그래프
    ax2.plot(rsi_series.index, rsi_series.values, color="orange")
    ax2.set_title("RSI")
    ax2.set_ylabel("RSI")
    ax2.axhline(y=70, color="red", linestyle="--")  # 과매수 기준선
    ax2.axhline(y=30, color="green", linestyle="--")  # 과매도 기준선
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    plt.show()


def main():
    # 샘플 데이터 생성 (실제 주식 데이터와 비슷한 패턴을 만들기 위해 random walk 사용)
    df = pd.read_parquet("D:/Workspace/DnS/data/NAVER_20190723_20240721.parquet")
    price_series = df["종가"]

    # RSI 계산
    rsi_values = calculate_rsi(price_series)

    # 가격과 RSI 시각화
    visualize_price_and_rsi(price_series, rsi_values)


if __name__ == "__main__":
    main()
