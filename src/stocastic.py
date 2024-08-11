import numpy as np
import pandas as pd
import talib


def stochastic_slow(
    data: pd.DataFrame,
    fastk_period: int = 14,
    slowk_period: int = 3,
    slowd_period: int = 3,
) -> pd.DataFrame:
    """
    Calculate the Stochastic Slow indicator using TA-Lib.

    :param data: pandas DataFrame with '고가', '저가', '종가' columns
    :param fastk_period: The fast %K period (default is 14)
    :param slowk_period: The slow %K period (default is 3)
    :param slowd_period: The slow %D period (default is 3)
    :return: pandas DataFrame with slowk and slowd columns
    """
    slowk, slowd = talib.STOCH(
        data["고가"],
        data["저가"],
        data["종가"],
        fastk_period=fastk_period,
        slowk_period=slowk_period,
        slowk_matype=0,
        slowd_period=slowd_period,
        slowd_matype=0,
    )

    return pd.DataFrame({"slowk": slowk, "slowd": slowd}, index=data.index)


def stochastic_fast(
    data: pd.DataFrame, fastk_period: int = 14, fastd_period: int = 3
) -> pd.DataFrame:
    """
    Calculate the Stochastic Fast indicator using TA-Lib.

    :param data: pandas DataFrame with '고가', '저가', '종가' columns
    :param fastk_period: The fast %K period (default is 14)
    :param fastd_period: The fast %D period (default is 3)
    :return: pandas DataFrame with fastk and fastd columns
    """
    fastk, fastd = talib.STOCHF(
        data["고가"],
        data["저가"],
        data["종가"],
        fastk_period=fastk_period,
        fastd_period=fastd_period,
        fastd_matype=0,
    )

    return pd.DataFrame({"fastk": fastk, "fastd": fastd}, index=data.index)


# 사용 예시
if __name__ == "__main__":
    # 샘플 데이터 생성 (실제 사용 시에는 이 부분을 실제 데이터 로드로 대체하세요)
    data = pd.read_parquet("C:/Users/yjahn/Desktop/DnS/data/NAVER_20190806_20240804.parquet")

    # Stochastic Slow 계산
    slow = stochastic_slow(data)
    print("Stochastic Slow:")
    print(slow)

    # Stochastic Fast 계산
    fast = stochastic_fast(data)
    print("\nStochastic Fast:")
    print(fast)
