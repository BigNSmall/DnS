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
from fractional_difference import fractional_difference
from vix import calculate_volatility

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
def prepare_data(file_path):
    df = pd.read_parquet(file_path)
    window_size = 5
    stride = 2
    df_list = create_time_windows(df, window_size, stride)

    features = {
        "vix": calculate_volatility(df, column="종가", window_size=window_size),
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
            f for f in os.listdir(directory_path) if f.endswith(".parquet")
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