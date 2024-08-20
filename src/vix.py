import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from utils.time_windowed_data import create_time_windows

# 한글 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"  # 실제 한글 폰트 경로로 변경해야 합니다
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams["font.family"] = font_prop.get_name()


def calculate_volatility(df: pd.DataFrame, column: str, window_size: int) -> np.ndarray:
    data = df[column].values
    return np.array(
        [np.std(data[i : i + window_size]) for i in range(len(df) - window_size + 1)]
    )


def plot_volatility(volatility_values: np.ndarray, title: str):
    """
    변동성 값의 시차별 변화를 시각화합니다.

    Parameters:
    - volatility_values: np.ndarray, 각 window의 시차별 변동성 값 (shape: [windows, lags])
    - title: str, 그래프 제목
    """
    num_windows, num_lags = volatility_values.shape

    plt.figure(figsize=(12, 6))

    # 각 시차에 대한 변동성의 평균값을 플롯
    mean_volatility = np.mean(volatility_values, axis=0)
    plt.plot(range(num_lags), mean_volatility, "r-", label="Mean Volatility")

    plt.xlabel("Lag")
    plt.ylabel("Volatility")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    df = pd.read_parquet(
        "C:/Users/yjahn/Desktop/DnS/data/NAVER_20190806_20240804.parquet"
    )

    # 시계열 데이터를 time window로 나누기
    window_size = 5  # window 크기 및 최대 시차
    stride = 2  # stride 크기
    df_list = create_time_windows(df, window_size, stride)

    print("종가에 대한 변동성 계산 결과:")
    volatility_values = calculate_volatility(
        df_list, column="종가", window_size=window_size
    )
    plot_volatility(
        volatility_values,
        f"Volatility by Lag (Window Size: {window_size}, Stride: {stride})",
    )


if __name__ == "__main__":
    main()
