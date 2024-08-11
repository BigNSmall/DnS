import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def div_each_before(time_series):
    """
    시계열 데이터를 입력받아 각 값을 이전 값으로 나누는 DIV_EACH_BEFORE 방식으로 변환합니다.

    :param time_series: pandas.Series, 시계열 데이터
    :return: pandas.Series, DIV_EACH_BEFORE 방식으로 변환된 데이터
    """
    div_each_before = time_series / time_series.shift(1)
    div_each_before.iloc[0] = 1
    return div_each_before


def main():
    # 예시 데이터 생성
    df = pd.read_parquet("C:/Users/yjahn/Desktop/DnS/data/NAVER_20190806_20240804.parquet")   

    time_series = df["종가"]

    # DIV_EACH_BEFORE 적용
    result = div_each_before(time_series)

    # 시각화
    plt.figure(figsize=(12, 8))

    # 원본 데이터 그래프
    plt.subplot(2, 1, 1)
    plt.plot(time_series.index, time_series.values, label="Original Data")
    plt.title("Original Time Series Data")
    plt.ylabel("Value")
    plt.legend()

    # 변환된 데이터 그래프
    plt.subplot(2, 1, 2)
    plt.plot(result.index, result.values, label="DIV_EACH_BEFORE", color="orange")
    plt.axhline(y=1, color="r", linestyle="--", label="No Change Line")
    plt.title("DIV_EACH_BEFORE Transformed Data")
    plt.ylabel("Ratio")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 통계 정보 출력
    print("원본 데이터 통계:")
    print(time_series.describe())
    print("\nDIV_EACH_BEFORE 변환 데이터 통계:")
    print(result.describe())


if __name__ == "__main__":
    main()
