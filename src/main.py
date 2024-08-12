import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils.time_windowed_data import create_time_windows
from acf import calculate_acf
from fractional_difference import fractional_difference

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def prepare_data():
    df = pd.read_parquet("D:/Workspace/DnS/data/NAVER_20190723_20240721.parquet")
    window_size = 5
    stride = 2
    df_list = create_time_windows(df, window_size, stride)

    features = {
        "ACF": calculate_acf(df_list, column="종가", window_size=window_size),
    }

    close_prices = df["종가"]
    d_values = [0.8, 0.5, 0.3]
    for d in d_values:
        frac_diffs = fractional_difference(close_prices, d)
        frac_diffs = create_time_windows(frac_diffs, window_size, stride)
        features[f"Frac_Diff_{d}"] = frac_diffs

    return features


def add_correlation(data1, data2):
    flat_data1 = (
        data1.ravel() if isinstance(data1, np.ndarray) else data1.values.ravel()
    )
    flat_data2 = (
        data2.ravel() if isinstance(data2, np.ndarray) else data2.values.ravel()
    )
    correlation = np.corrcoef(flat_data1, flat_data2)[0, 1]
    return correlation


def calculate_correlation(features):
    labels = list(features.keys())
    n = len(labels)
    corr_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1):  # 하삼각행렬만 계산
            corr_matrix[i, j] = add_correlation(
                features[labels[i]], features[labels[j]]
            )

    # 대각선 아래쪽만 남기고 나머지는 NaN으로 설정
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    corr_matrix[mask] = np.nan

    return pd.DataFrame(corr_matrix, index=labels, columns=labels)


def main():
    st.title("Feature Correlation Heatmap (Lower Triangle)")

    features = prepare_data()
    corr_df = calculate_correlation(features)

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_df, dtype=bool), k=1)
    sns.heatmap(
        corr_df,
        mask=mask,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        ax=ax,
        cbar_kws={"label": "Correlation"},
    )
    plt.title("Correlation Heatmap (Lower Triangle)")
    st.pyplot(fig)

    st.write("Correlation Matrix (Lower Triangle):")
    st.dataframe(corr_df)


if __name__ == "__main__":
    main()
