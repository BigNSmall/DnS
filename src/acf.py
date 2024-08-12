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


def calculate_acf(df: pd.DataFrame, column: str, window_size: int) -> np.ndarray:
    """
    시간 창이 적용된 DataFrame의 지정된 열에 대해 자기상관함수(ACF)를 계산합니다.

    :param df: create_time_windows 함수로 생성된 시간 창 DataFrame
    :param column: ACF를 계산할 기준 열의 이름 (예: "종가_t-0")
    :param window_size: ACF를 계산할 최대 시차
    :return: 계산된 ACF 값을 포함하는 2D NumPy 배열 (각 행이 하나의 시간 창에 대한 ACF 값)
    """
    acf_values_list = []

    # 각 시간 창에 대해 ACF 계산
    for i in range(len(df)):
        # 현재 시간 창의 데이터 추출
        window_data = df.iloc[i, :window_size]

        # ACF 계산
        acf_values = acf(window_data, nlags=window_size - 1, fft=False)

        acf_values_list.append(acf_values)

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
