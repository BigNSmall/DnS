import glob
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


def calculate_indicators(df, window_size):
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

    indicators = {
        "ACF": calculate_acf(df["종가"], window_size=window_size),  # (T, window_size)
        "Buffett": calculate_buffett_index(df["종가"], "KOR"),  # (T,)
        "DE": demartini_index(df["종가"]),  # (T,)
        "DEB": div_each_before(df["종가"]),  # (T,)
        "FracDiff": fractional_difference(df["종가"], 0.3),  # (T,)
        "PivotPoints": calculate_pivot_points(
            df["고가"], df["저가"], df["종가"]
        ),  # (T, 7)
        "SN": sonar_indicator(df, window_size=14),  # (T,)
        "STFA": stochastic_fast(df)["fastd"],  # (T,)
        "STSL": stochastic_slow(df)["slowd"],  # (T,)
        "TimeDelay": time_delay_embedding(df["종가"], 60, 1)["t-0"],  # (T,)
        "CalVIX": calculate_vix(df["종가"], window_size),  # (T,)
        "Williams": williams_r(df, 5),  # (T,)
    }

    # Ensure all indicators are 2D numpy arrays
    for key, value in indicators.items():
        if isinstance(value, pd.DataFrame):
            indicators[key] = value.values
        elif isinstance(value, pd.Series):
            indicators[key] = value.values.reshape(-1, 1)
        elif isinstance(value, np.ndarray):
            if value.ndim == 1:
                indicators[key] = value.reshape(-1, 1)
            elif value.ndim > 2:
                raise ValueError(f"Indicator {key} has more than 2 dimensions")
        else:
            raise ValueError(f"Unexpected type for indicator {key}: {type(value)}")

    return indicators


class SlidingWindowDataset(Dataset):
    def __init__(self, df, window_size=5, stride=2):
        self.df = df
        self.window_size = window_size
        self.stride = stride

        # Calculate indicators
        self.indicators = calculate_indicators(df, window_size)

        # Combine all indicators into a single numpy array
        self.combined_data = np.concatenate(
            [v for v in self.indicators.values()], axis=1
        )

        # Remove any rows with NaN values
        self.combined_data = self.combined_data[
            ~np.isnan(self.combined_data).any(axis=1)
        ]

        # Calculate the number of valid windows
        self.num_windows = (len(self.combined_data) - window_size) // stride + 1

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.window_size

        window_data = self.combined_data[start_idx:end_idx]

        # Check if the window contains any NaN values
        if np.isnan(window_data).any():
            # If NaN values are found, move to the next valid window
            return self.__getitem__(idx + 1)

        # Apply MinMax scaling per column
        scaled_window = self.scale_window(window_data)

        return scaled_window

    def scale_window(self, window):
        """
        각 컬럼별로 MinMax 스케일링을 적용합니다.
        """
        scaler = StandardScaler()
        scaled_window = np.zeros_like(window)
        for i in range(window.shape[1]):
            scaled_window[:, i] = scaler.fit_transform(
                window[:, i].reshape(-1, 1)
            ).flatten()
        return scaled_window


class MultiFileSlidingWindowDataset(Dataset):
    def __init__(self, file_pattern, window_size=5, stride=2):
        self.file_list = sorted(glob.glob(file_pattern))
        self.window_size = window_size
        self.stride = stride
        self.file_index = []
        self.cumulative_windows = [0]

        for i, file_path in enumerate(self.file_list[:10]):
            df = pd.read_parquet(file_path)
            dataset = SlidingWindowDataset(df, window_size, stride)
            num_windows = len(dataset)
            self.file_index.extend([i] * num_windows)
            self.cumulative_windows.append(self.cumulative_windows[-1] + num_windows)

    def __len__(self):
        return self.cumulative_windows[-1]

    def __getitem__(self, idx):
        file_idx = self.file_index[idx]
        file_path = self.file_list[file_idx]
        local_idx = idx - self.cumulative_windows[file_idx]

        df = pd.read_parquet(file_path)
        dataset = SlidingWindowDataset(df, self.window_size, self.stride)
        return dataset[local_idx]


def data_generator(file_pattern, window_size=5, stride=2, batch_size=32):
    file_list = sorted(glob.glob(file_pattern))
    for file_path in file_list:
        df = pd.read_parquet(file_path)
        dataset = SlidingWindowDataset(df, window_size, stride)
        for i in range(0, len(dataset), batch_size):
            yield [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]


# Example usage:
if __name__ == "__main__":
    file_pattern = "./data/*.parquet"  # 파일 패턴을 적절히 수정하세요.

    # 방법 1: MultiFileSlidingWindowDataset 사용
    multi_dataset = MultiFileSlidingWindowDataset(file_pattern, window_size=5, stride=2)
    print(f"Total dataset length: {len(multi_dataset)}")
    print(f"Sample window shape: {multi_dataset[0].shape}")
    print(f"Sample window data:\n{multi_dataset[0]}")

    # 방법 2: 제너레이터 사용
    for batch in data_generator(file_pattern, window_size=5, stride=2, batch_size=32):
        print(f"Batch size: {len(batch)}")
        print(f"Sample window shape: {batch[0].shape}")
        print(f"Sample window data:\n{batch[0]}")
        break  # 예시를 위해 첫 번째 배치만 출력합니다.
