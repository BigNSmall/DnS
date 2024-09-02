import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from indicators.acf import calculate_acf
from indicators.buffett import calculate_buffett_index
from indicators.deMartini import demartini_index
from indicators.div_each_before import div_each_before
from indicators.fractional_difference import fractional_difference
from indicators.pivot import calculate_pivot_points
from indicators.sonar import sonar_indicator
from indicators.stocastic import stochastic_fast, stochastic_slow
from indicators.time_delay import time_delay_embedding
from indicators.vix import calculate_vix
from indicators.williams import williams_r


def calculate_indicators(df, window_size):
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


class SlidingWindowDataset(Dataset):
    def __init__(self, df, window_size=5, stride=2):
        self.df = df
        self.window_size = window_size
        self.stride = stride
        self.valid_indices = self._get_valid_indices()

        # Apply min-max normalization to each column
        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(self.df)

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

        window = self.normalized_data[start_idx:end_idx]

        # Convert to PyTorch tensor
        window_tensor = torch.FloatTensor(window)

        return {"window": window_tensor}

    def inverse_transform(self, normalized_data):
        return self.scaler.inverse_transform(normalized_data)


# Usage example:
# 파라미터 설정
window_size = 5
stride = 3
batch_size = 64
num_epochs = 100
learning_rate = 0.0001
test_size = 0.2


df = pd.read_parquet(
    r"D:\Workspace\DnS\data\AJ네트웍스_20190825_20240825.parquet"
)  # Load your DataFrame
indicators = calculate_indicators(df, window_size)
combined_df = combine_indicators(indicators, df)

# 데이터 분할
train_df, test_df = train_test_split(combined_df, test_size=test_size, shuffle=True)

# 데이터셋 및 DataLoader 생성
train_dataset = SlidingWindowDataset(train_df, window_size=window_size, stride=stride)
test_dataset = SlidingWindowDataset(test_df, window_size=window_size, stride=stride)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

print(train_loader[0])
