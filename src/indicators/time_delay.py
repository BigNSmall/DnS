import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


import pandas as pd


def time_delay_embedding(
    series: pd.Series, delay: int, embedding_dimension: int
) -> pd.DataFrame:
    if not isinstance(series, pd.Series):
        raise ValueError("Input must be a pandas Series")

    if delay <= 0 or embedding_dimension <= 0:
        raise ValueError("Delay and embedding dimension must be positive integers")

    n = len(series)
    if n < delay * (embedding_dimension - 1) + 1:
        raise ValueError(
            "Series is too short for the given delay and embedding dimension"
        )

    embedded_data = []
    valid_indices = []

    for i in range(n - delay * (embedding_dimension - 1)):
        embedded_vector = [
            series.iloc[i + j * delay] for j in range(embedding_dimension)
        ]
        embedded_data.append(embedded_vector)
        valid_indices.append(series.index[i + delay * (embedding_dimension - 1)])

    column_names = [f"t-{j*delay}" for j in range(embedding_dimension - 1, -1, -1)]
    df = pd.DataFrame(embedded_data, columns=column_names, index=valid_indices)

    # Ensure the index is a DateTimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    return df


def main():
    # 파일에서 데이터 읽기
    data = pd.read_parquet("D:/Workspace/DnS/data/흥국화재_20190723_20240721.parquet")

    # 데이터프레임 구조 확인
    print("Columns in the dataframe:", data.columns)
    print("Index of the dataframe:", data.index)

    # '날짜' 열이 이미 인덱스인지 확인
    if "날짜" not in data.columns and data.index.name != "날짜":
        # '날짜' 열이 없고 인덱스도 아니라면, 첫 번째 열을 날짜로 가정
        data.index = pd.to_datetime(data.index)
        data.index.name = "날짜"
    elif "날짜" in data.columns:
        # '날짜' 열이 있다면 인덱스로 설정
        data["날짜"] = pd.to_datetime(data["날짜"])
        data.set_index("날짜", inplace=True)
    else:
        # 이미 '날짜'가 인덱스라면 datetime으로 변환
        data.index = pd.to_datetime(data.index)

    print("Updated index of the dataframe:", data.index)

    # Time delay embedding 수행
    delay = 30
    embedding_dimension = 3
    embedded_data = time_delay_embedding(data["종가"], delay, embedding_dimension)

    # 3D 산점도 그리기
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        embedded_data.iloc[:, 0],
        embedded_data.iloc[:, 1],
        embedded_data.iloc[:, 2],
        c=range(len(embedded_data)),
        cmap="viridis",
    )

    ax.set_xlabel(f"t")
    ax.set_ylabel(f"t-{delay}")
    ax.set_zlabel(f"t-{2*delay}")
    ax.set_title(
        f"Stock Price Time Delay Embedding (delay={delay}, dim={embedding_dimension})"
    )

    cbar = fig.colorbar(scatter)
    cbar.set_label("Time")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
