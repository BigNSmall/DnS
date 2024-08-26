import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def stochastic_slow(
    data: pd.DataFrame,
    fastk_period: int = 14,
    slowk_period: int = 3,
    slowd_period: int = 3,
    window_size=5,
) -> pd.DataFrame:
    """
    Calculate the Stochastic Slow indicator using Pandas and NumPy.

    :param data: pandas DataFrame with '고가', '저가', '종가' columns
    :param fastk_period: The fast %K period (default is 14)
    :param slowk_period: The slow %K period (default is 3)
    :param slowd_period: The slow %D period (default is 3)
    :return: pandas DataFrame with slowk and slowd columns
    """
    low_min = data["저가"].rolling(window=fastk_period, min_periods=1).min()
    high_max = data["고가"].rolling(window=fastk_period, min_periods=1).max()

    fastk = 100 * ((data["종가"] - low_min) / (high_max - low_min))
    slowk = fastk.rolling(window=slowk_period, min_periods=1).mean()
    slowd = slowk.rolling(window=slowd_period, min_periods=1).mean()
    #slowk = slowk[:window_size]
    return {
        "slowd": slowd,
        "slowk": slowk,
    }
    #return pd.DataFrame({"slowd": slowd, "slowk": slowk}, index=data.index)



def stochastic_fast(
    data: pd.DataFrame, fastk_period: int = 14, fastd_period: int = 3, window_size=5
) -> pd.DataFrame:
    """
    Calculate the Stochastic Fast indicator using Pandas and NumPy.

    :param data: pandas DataFrame with '고가', '저가', '종가' columns
    :param fastk_period: The fast %K period (default is 14)
    :param fastd_period: The fast %D period (default is 3)
    :return: pandas DataFrame with fastk and fastd columns
    """
    low_min = data["저가"].rolling(window=fastk_period, min_periods=1).min()
    high_max = data["고가"].rolling(window=fastk_period, min_periods=1).max()

    fastk = 100 * ((data["종가"] - low_min) / (high_max - low_min))
    fastd = fastk.rolling(window=fastd_period, min_periods=1).mean()

    fastd = fastd[:window_size]
    #fastd = fastd[:window_size]
    #fastk = fastk[:window_size]
    return {"fastk": fastk, "fastd": fastd}
    #return pd.DataFrame({"fastk": fastk, "fastd": fastd}, index=data.index)


# 사용 예시
if __name__ == "__main__":
    # 샘플 데이터 생성 (실제 사용 시에는 이 부분을 실제 데이터 로드로 대체하세요)
    data = pd.read_parquet(
        "C:/Users/yjahn/Desktop/DnS/data/NAVER_20190806_20240804.parquet"
    )

    # Stochastic Slow 계산
    slow = stochastic_slow(data)
    print("Stochastic Slow:")
    print(slow)

    # Stochastic Fast 계산
    fast = stochastic_fast(data)
    print("\nStochastic Fast:")
    print(fast)

    # 시각화
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Stochastic Slow 그래프
    ax1.plot(slow, label="Slow %K")
    ax1.set_title("Stochastic Slow")
    ax1.set_ylabel("Value")
    ax1.legend()
    ax1.grid(True)

    # Stochastic Fast 그래프
    ax2.plot(fast, label="Fast %K")
    ax2.set_title("Stochastic Fast")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Value")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
