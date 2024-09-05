import pandas as pd
import numpy as np
from torch.utils.data import Dataset


def create_time_windows(df, window_size, stride):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("window_size must be a positive integer")
    if not isinstance(stride, int) or stride <= 0:
        raise ValueError("stride must be a positive integer")

    result_dict = {}
    for column in df.columns:
        windows = {
            df.index[i + window_size - 1]: df[column].iloc[i : i + window_size].values
            for i in range(0, len(df) - window_size + 1, stride)
        }
        window_df = pd.DataFrame.from_dict(windows, orient="index")
        window_df.columns = [
            f"{column}_t-{window_size-i-1}" for i in range(window_size)
        ]
        result_dict[column] = window_df

    return result_dict


def calculate_indicators(df, window_size):
    from acf import calculate_acf
    from buffett import calculate_buffett_index
    from deMartini import demartini_index
    from div_each_before import div_each_before
    from fractional_difference import fractional_difference
    from pivot import calculate_pivot_points
    from sonar import sonar_indicator
    from stocastic import stochastic_fast, stochastic_slow
    from time_delay import time_delay_embedding
    from vix import calculate_vix
    from williams import williams_r

    indicators = {
        "ACF": calculate_acf(df["종가"], window_size=window_size),
        "Buffett": calculate_buffett_index(df["종가"], "KOR"),
        "DE": demartini_index(df["종가"]),
        "DEB": div_each_before(df["종가"]),
        "FracDiff": fractional_difference(df["종가"], 0.3),
        "PivotPoints": calculate_pivot_points(df["고가"], df["저가"], df["종가"]),
        "SN": sonar_indicator(df, window_size=14),
        "STFA": stochastic_fast(df)["fastd"],
        "STSL": stochastic_slow(df)["slowd"],
        "TimeDelay": time_delay_embedding(df["종가"], 60, 1)["t-0"],
        "CalVIX": calculate_vix(df["종가"], window_size),
        "Williams": williams_r(df, 5),
    }

    return indicators


def combine_indicators(indicators, df_original):
    column_mapping = {
        "acf": "Autocorrelation_Function",
        "buffett": "Buffett_Indicator",
        "de": "DeMartini_Indicator",
        "deb": "Divide_Each_Before",
        "fracdiff": "Fractional_Differentiation",
        "pivot_high": "Pivot_Point_High",
        "pivot_low": "Pivot_Point_Low",
        "pivot_close": "Pivot_Point_Close",
        "sonar": "Sonar_Indicator",
        "fastd": "Stochastic_Fast_%D",
        "slowd": "Stochastic_Slow_%D",
        "t-0": "Time_Delay_Embedding",
        "vix": "Volatility_Index",
        "williams_r": "Williams_%R",
    }

    result = pd.DataFrame(index=df_original.index)

    for indicator_name, indicator_data in indicators.items():
        if isinstance(indicator_data, pd.DataFrame):
            renamed_columns = {
                col: f"{indicator_name}_{column_mapping.get(col, col)}"
                for col in indicator_data.columns
            }
            indicator_data = indicator_data.rename(columns=renamed_columns)
        elif isinstance(indicator_data, pd.Series):
            new_name = f"{indicator_name}_{column_mapping.get(indicator_data.name, indicator_data.name)}"
            indicator_data = indicator_data.rename(new_name)

        result = pd.concat([result, indicator_data], axis=1)

    return result


class SlidingWindowDataLoader(Dataset):
    def __init__(self, df, window_size=5, stride=2):
        self.df = df
        self.window_size = window_size
        self.stride = stride
        self.valid_indices = self._get_valid_indices()

    def _get_valid_indices(self):
        return [
            i
            for i in range(0, len(self.df) - self.window_size + 1, self.stride)
            if not np.isnan(self.df.iloc[i : i + self.window_size].values).any()
        ]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        if idx >= len(self.valid_indices):
            raise IndexError("Index out of range")

        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.window_size

        window = self.df.iloc[start_idx:end_idx].values

        return {"window": window, "timestamp": self.df.index[end_idx - 1]}


def main():
    # Load data
    df = pd.read_parquet(r"D:\Workspace\DnS\data\AJ네트웍스_20190825_20240825.parquet")

    # Set parameters
    window_size = 5
    stride = 2

    # Calculate indicators
    indicators = calculate_indicators(df, window_size)

    # Combine indicators
    combined_df = combine_indicators(indicators, df)

    # Print results
    print(combined_df.head().to_markdown())
    print(f"Shape of combined DataFrame: {combined_df.shape}")

    # Create dataset
    dataset = SlidingWindowDataLoader(
        combined_df, window_size=window_size, stride=stride
    )

    print(f"\nTotal number of valid samples: {len(dataset)}")

    # Print sample data
    for i in range(len(dataset)):  # Print first 3 samples
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"Window shape: {sample['window'].shape}")
        print(f"Timestamp: {sample['timestamp']}")
        print("-" * 25)


if __name__ == "__main__":
    main()
