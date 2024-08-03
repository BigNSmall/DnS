import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf
from typing import List
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"  # 실제 한글 폰트 경로로 변경해야 합니다
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams["font.family"] = font_prop.get_name()


def create_time_windows(
    df: pd.DataFrame, window: int, stride: int
) -> List[pd.DataFrame]:
    values = df.values
    indices = np.arange(len(df) - window + 1, step=stride)
    windowed_values = np.array([values[i : i + window] for i in indices])
    return [
        pd.DataFrame(window_data, index=df.index[i : i + window], columns=df.columns)
        for i, window_data in zip(indices, windowed_values)
    ]


def calculate_acf(
    data_list: List[pd.DataFrame], column: str, window_size: int
) -> np.ndarray:
    acf_values_list = []

    for df in data_list:
        series = df[column]
        acf_values = acf(series, nlags=window_size - 1, alpha=0.05, fft=False)
        acf_values_list.append(acf_values[0])  # ACF 값만 저장 (신뢰 구간 제외)

    return np.array(acf_values_list)


def plot_acf_range(acf_values: np.ndarray, title: str):
    """
    ACF 값의 범위를 시각화합니다.

    Parameters:
    - acf_values: np.ndarray, 각 window의 ACF 값 (shape: [windows, lags])
    - title: str, 그래프 제목
    """
    num_lags = acf_values.shape[1]

    plt.figure(figsize=(12, 6))

    # 각 lag에 대한 최소값, 최대값, 평균값 계산
    min_values = np.min(acf_values, axis=0)
    max_values = np.max(acf_values, axis=0)
    mean_values = np.mean(acf_values, axis=0)

    # 범위 플롯 그리기
    plt.fill_between(range(num_lags), min_values, max_values, alpha=0.3, label="Range")
    plt.plot(range(num_lags), mean_values, "r-", label="Mean")

    plt.xlabel("Lag")
    plt.ylabel("ACF")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

    # 평균 ACF 출력
    for i, acf_value in enumerate(mean_values):
        print(f"Lag {i}: {acf_value:.4f}")


def main():
    df = pd.read_parquet("D:/Workspace/DnS/data/NAVER_20190723_20240721.parquet")

    # 시계열 데이터를 time window로 나누기
    window_size = 5  # window 크기
    stride = 2  # stride 크기
    df_list = create_time_windows(df, window_size, stride)

    print("종가에 대한 ACF 계산 결과:")
    acf_values = calculate_acf(df_list, column="종가", window_size=window_size)
    plot_acf_range(
        acf_values,
        f"Autocorrelation Function (ACF) for 종가 (Window Size: {window_size}, Stride: {stride})",
    )


if __name__ == "__main__":
    main()
