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
    """
    시간 창이 적용된 DataFrame의 지정된 열에 대해 변동성(VIX 지수와 유사)을 계산합니다.

    :param df: create_time_windows 함수로 생성된 시간 창 DataFrame
    :param column: 변동성을 계산할 기준 열의 이름 (예: "종가_t-0")
    :param window_size: 변동성을 계산할 시간 창 크기 및 최대 시차
    :return: 계산된 변동성 값을 포함하는 2D NumPy 배열 (각 행이 하나의 시간 창에 대한 변동성 값)
    """
    volatility_values = np.zeros((len(df), window_size))

    # 각 시간 창에 대해 변동성 계산
    for i in range(len(df)):
        # 현재 시간 창의 데이터 추출
        window_data = df.iloc[i, :window_size] # 특정 열의 데이터만 추출하고 결측값 제거

        # 각 시차에 대해 변동성 계산
        for lag in range(window_size):
            if len(window_data) > lag:
                lagged_data = window_data[lag:]  # 시차를 고려한 데이터 추출
                # 변동성 (표준편차) 계산
                volatility = np.std(lagged_data)
                volatility_values[i, lag] = volatility

    return volatility_values

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
    df = pd.read_parquet("C:/Users/yjahn/Desktop/DnS/data/NAVER_20190806_20240804.parquet")
    
    # 시계열 데이터를 time window로 나누기
    window_size = 5  # window 크기 및 최대 시차
    stride = 2  # stride 크기
    df_list = create_time_windows(df, window_size, stride)

    print("종가에 대한 변동성 계산 결과:")
    volatility_values = calculate_volatility(df_list, column="종가", window_size=window_size)
    plot_volatility(
        volatility_values,
        f"Volatility by Lag (Window Size: {window_size}, Stride: {stride})",
    )

if __name__ == "__main__":
    main()