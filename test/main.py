import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.time_windowed_data import create_time_windows
from acf import calculate_acf
from div_each_before import div_each_before
from time_delay import time_delay_embedding
from stocastic import stochastic_fast, stochastic_slow
from fractional_difference import fractional_difference
from vix import calculate_vix
from williams import williams_r
from buffett import calculate_buffett_index
from deMartini import demartini_index
from pivot import calculate_pivot_points
from sonar import sonar_indicator

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def prepare_data(file_path):
    df = pd.read_parquet(file_path)
    window_size = 5
    stride = 2
    df_list = create_time_windows(df, window_size, stride)

    features = {
        "vix": calculate_vix(df["종가"]),
        "williams": williams_r(df, window_size, 5),
        "ACF": calculate_acf(df_list, column="종가", window_size=window_size),
        "div_each_before": div_each_before(df["종가"]),
        "time delay": time_delay_embedding(df["종가"], 5, 3),
        "stocastic fast": stochastic_fast(df)["fastk"],
        "stocastic slow": stochastic_slow(df)["slowk"],
        "buffet index kor": calculate_buffett_index(df["종가"], "KOR"),
        # "buffet index usa": calculate_buffett_index(df["종가"], "USA"),
        "demartini index": demartini_index(df["종가"], period=5),
        "pivot": calculate_pivot_points(df["고가"], df["저가"], df["종가"]),
        "sonar": sonar_indicator(df, window_size=5),
    }

    close_prices = df["종가"]
    d_values = [0.8, 0.3]
    for d in d_values:
        frac_diffs = fractional_difference(close_prices, d)
        frac_diffs = create_time_windows(frac_diffs, window_size, stride)
        features[f"Frac_Diff_{d}"] = frac_diffs

    # Ensure all features have the same length
    min_length = min(len(v) for v in features.values())
    features = {k: v[:min_length] for k, v in features.items()}

    return features


def add_correlation(data1, data2):
    flat_data1 = (
        data1.ravel() if isinstance(data1, np.ndarray) else data1.values.ravel()
    )
    flat_data2 = (
        data2.ravel() if isinstance(data2, np.ndarray) else data2.values.ravel()
    )

    # Ensure both arrays have the same length
    min_length = min(len(flat_data1), len(flat_data2))
    flat_data1 = flat_data1[:min_length]
    flat_data2 = flat_data2[:min_length]

    correlation = np.corrcoef(flat_data1, flat_data2)[0, 1]
    return correlation


def calculate_correlation(features_list):
    labels = list(features_list[0].keys())
    n = len(labels)
    corr_matrices = []

    def calculate_single_correlation(features):
        corr_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1):
                corr_matrix[i, j] = add_correlation(
                    features[labels[i]], features[labels[j]]
                )
        return corr_matrix

    with ThreadPoolExecutor() as executor:
        corr_matrices = list(executor.map(calculate_single_correlation, features_list))

    avg_corr_matrix = np.mean(corr_matrices, axis=0)

    mask = np.triu(np.ones_like(avg_corr_matrix, dtype=bool), k=1)
    avg_corr_matrix[mask] = np.nan

    return pd.DataFrame(avg_corr_matrix, index=labels, columns=labels)


def main():
    st.title("Feature Correlation Heatmap (Lower Triangle)")

    directory_path = st.text_input(
        "Enter the directory path containing .parquet files:",
    )

    if st.button("Calculate Correlation"):
        start_time = time.time()

        if not os.path.isdir(directory_path):
            st.error("The provided path is not a valid directory.")
            return

        parquet_files = [
            f for f in os.listdir(directory_path)[:10] if f.endswith(".parquet")
        ]

        if not parquet_files:
            st.error("No .parquet files found in the specified directory.")
            return

        st.info(f"Found {len(parquet_files)} .parquet files in the directory.")

        features_list = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        def process_file(file):
            file_path = os.path.join(directory_path, file)
            return prepare_data(file_path)

        with ThreadPoolExecutor() as executor:
            future_to_file = {
                executor.submit(process_file, file): file for file in parquet_files
            }
            for i, future in enumerate(as_completed(future_to_file)):
                file = future_to_file[future]
                try:
                    features = future.result()
                    features_list.append(features)
                except Exception as exc:
                    st.error(f"{file} generated an exception: {exc}")
                status_text.text(f"Processing file {i+1}/{len(parquet_files)}: {file}")
                progress_bar.progress((i + 1) / len(parquet_files))

        status_text.text("Calculating correlation matrix...")
        corr_df = calculate_correlation(features_list)

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
        plt.title("Average Correlation Heatmap (Lower Triangle)")
        st.pyplot(fig)

        st.write("Average Correlation Matrix (Lower Triangle):")
        st.dataframe(corr_df)

        end_time = time.time()
        execution_time = end_time - start_time
        st.success(f"Calculation completed in {execution_time:.2f} seconds.")

        progress_bar.empty()
        status_text.empty()


if __name__ == "__main__":
    main()
