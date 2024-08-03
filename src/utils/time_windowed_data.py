import pandas as pd
import numpy as np
from typing import List


def create_time_windows(
    df: pd.DataFrame, window: int, stride: int
) -> List[pd.DataFrame]:
    """
    주식 데이터(시계열)를 time window 기법으로 나누는 최적화된 함수

    :param df: 입력 DataFrame (시계열 데이터)
    :param window: window 크기 (정수)
    :param stride: stride 크기 (정수)
    :return: 분할된 DataFrame들의 리스트
    """
    # NumPy 배열로 변환
    values = df.values

    # 인덱스 계산
    indices = np.arange(len(df) - window + 1, step=stride)

    # 한 번에 모든 window 생성
    windowed_values = np.array([values[i : i + window] for i in indices])

    # DataFrame 리스트로 변환
    return [
        pd.DataFrame(window_data, index=df.index[i : i + window], columns=df.columns)
        for i, window_data in zip(indices, windowed_values)
    ]
