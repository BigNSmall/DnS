import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf
from typing import List
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from utils.time_windowed_data import create_time_windows

# 한글 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"  # 실제 한글 폰트 경로로 변경해야 합니다
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams["font.family"] = font_prop.get_name()


def calculate_acf(
    time_series: pd.Series | pd.DataFrame, window_size: int = None
) -> pd.DataFrame:
    # 입력이 DataFrame인 경우 Series로 변환
    if isinstance(time_series, pd.DataFrame):
        time_series = time_series.iloc[:, 0]

    # window_size가 지정되지 않은 경우, 시계열 길이의 절반으로 설정
    if window_size is None:
        window_size = len(time_series) // 2

    # ACF 계산
    acf_values = acf(time_series, nlags=window_size - 1, fft=False)

    # 결과를 DataFrame으로 변환
    result = pd.DataFrame(
        np.tile(acf_values, (len(time_series), 1)),
        columns=[f"lag_{i}" for i in range(window_size)],
    )

    return result


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


def main():
    df = pd.read_parquet(
        "C:/Users/yjahn/Desktop/DnS/data/NAVER_20190806_20240804.parquet"
    )

    # 시계열 데이터를 time window로 나누기
    window_size = 5  # window 크기
    stride = 2  # stride 크기
    df_list = create_time_windows(df, window_size, stride)

    print("종가에 대한 ACF 계산 결과:")
    acf_values = calculate_acf(df_list, column="종가", window_size=window_size)
    print(acf_values)
    plot_acf_range(
        acf_values,
        f"Autocorrelation Function (ACF) for 종가 (Window Size: {window_size}, Stride: {stride})",
    )


if __name__ == "__main__":
    main()
