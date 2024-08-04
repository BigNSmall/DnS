import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def fractional_difference(series, d):
    """
    주어진 시계열에 대해 Fractional Difference를 계산합니다.

    :param series: pandas Series 객체, 입력 시계열 데이터
    :param d: float, 차분 차수 (0 < d < 1)
    :return: pandas Series, Fractional Difference가 적용된 시계열
    """

    # 가중치 계산
    def weights(d, size):
        w = [1.0]
        for k in range(1, size):
            w.append(-w[-1] * (d - k + 1) / k)
        return np.array(w)

    # 시계열의 길이
    T = len(series)

    # 가중치 계산 (최대 T-1개)
    w = weights(d, T)

    # Fractional Difference 계산
    res = pd.Series(index=series.index)
    for t in range(T):
        res.iloc[t] = np.sum(w[: t + 1][::-1] * series.iloc[: t + 1])

    return res


import pandas as pd
import matplotlib.pyplot as plt
from fractional_difference import fractional_difference


def main():
    # 데이터 로드
    df = pd.read_parquet("D:/Workspace/DnS/data/NAVER_20190723_20240721.parquet")

    # 종가 데이터 추출
    close_prices = df["종가"]

    # 다양한 d 값에 대해 Fractional Difference 적용
    d_values = [0.8, 0.5, 0.3]
    frac_diffs = {d: fractional_difference(close_prices, d) for d in d_values}

    # 그래프 생성
    plt.figure(figsize=(15, 10))

    # 원본 종가 그래프
    plt.subplot(2, 1, 1)
    plt.plot(
        close_prices.index, close_prices, label="Original Close Price", color="blue"
    )
    plt.title("NAVER Stock Close Price")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Fractional Difference 그래프
    plt.subplot(2, 1, 2)
    colors = ["red", "green", "purple"]
    for d, color in zip(d_values, colors):
        plt.plot(
            frac_diffs[d].index,
            frac_diffs[d],
            label=f"Fractional Diff (d={d})",
            color=color,
        )

    plt.title("Fractional Difference of NAVER Stock Close Price")
    plt.xlabel("Date")
    plt.ylabel("Fractional Difference")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.show()

    # 기본 통계량 출력
    print("Original Close Price - Basic Statistics:")
    print(close_prices.describe())

    for d in d_values:
        print(f"\nFractional Difference (d={d}) - Basic Statistics:")
        print(frac_diffs[d].describe())


if __name__ == "__main__":
    main()
