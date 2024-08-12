import pandas as pd
import numpy as np
from typing import List, Union


def create_time_windows(
    data: Union[pd.DataFrame, pd.Series], window: int, stride: int
) -> pd.DataFrame:
    """
    주식 데이터(시계열)를 time window 기법으로 나누는 최적화된 함수

    :param data: 입력 DataFrame 또는 Series (시계열 데이터)
    :param window: window 크기 (정수)
    :param stride: stride 크기 (정수)
    :return: 분할된 데이터를 포함하는 DataFrame
    """
    # Series를 DataFrame으로 변환 (필요한 경우)
    if isinstance(data, pd.Series):
        df = data.to_frame()
    else:
        df = data

    # NumPy 배열로 변환
    values = df.values

    # 인덱스 계산
    indices = np.arange(len(df) - window + 1, step=stride)

    # 한 번에 모든 window 생성
    windowed_values = np.array([values[i : i + window] for i in indices])

    # 결과를 저장할 빈 DataFrame 생성
    result_df = pd.DataFrame(index=indices)

    # window 데이터를 DataFrame에 추가
    for i in range(window):
        if isinstance(data, pd.Series):
            col_name = f"{data.name}_t-{window-i-1}" if data.name else f"t-{window-i-1}"
            result_df[col_name] = windowed_values[:, i]
        else:
            for col in df.columns:
                result_df[f"{col}_t-{window-i-1}"] = windowed_values[
                    :, i, df.columns.get_loc(col)
                ]

    return result_df
