import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from ta.volatility import BollingerBands

def calculate_vix(series: pd.Series, window: int = 30) -> pd.Series:
    """
    주어진 pandas Series에서 VIX와 유사한 지표를 계산하는 함수.

    Parameters:
    series (pd.Series): 입력 데이터 시리즈 (예: 주가)
    window (int): 변동성을 계산할 기간 (기본값: 30일)

    Returns:
    pd.Series: VIX와 유사한 지표 시리즈
    """
    # 로그 수익률 계산
    log_returns = np.log(series / series.shift(1)).dropna()
    
    # 30일 이동 표준 편차 계산
    rolling_std = log_returns.rolling(window=window).std()
    
    # 연율화된 변동성 계산 (표준 편차 * sqrt(252))
    annualized_volatility = rolling_std * np.sqrt(252)
    
    # VIX와 유사한 지표 생성 (100을 곱해서 %로 표시)
    vix_like_index = annualized_volatility * 100
    
    return vix_like_index

def plot_vix_and_prices(price_series: pd.Series, vix_series: pd.Series):
    """
    주가 데이터와 VIX와 유사한 지표를 시각화하는 함수.

    Parameters:
    price_series (pd.Series): 주가 데이터 시리즈
    vix_series (pd.Series): VIX와 유사한 지표 시리즈
    """
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # 주가 데이터 플롯
    ax1.plot(price_series, color='blue', label='Price')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # VIX 데이터 플롯 (이중 축 사용)
    ax2 = ax1.twinx()
    ax2.plot(vix_series, color='red', label='VIX-like Index')
    ax2.set_ylabel('VIX-like Index', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # 그래프 제목과 범례 설정
    plt.title('Price and VIX-like Index')
    fig.tight_layout()
    fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.9))

    plt.show()

# 예시 데이터 사용
if __name__ == "__main__":
    # 가상 주가 데이터 생성
    df = pd.read_parquet("C:/Users/yjahn/Desktop/DnS/data/NAVER_20190806_20240804.parquet")
    price_series = df["종가"]

    # VIX와 유사한 지표 계산
    vix_series = calculate_vix(price_series)
    
    # 시각화
    plot_vix_and_prices(price_series, vix_series)