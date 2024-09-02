import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader


class SlidingWindowDataLoader(Dataset):
    def __init__(self, dir_path: str, window_size: int = 5, stride: int = 2):
        self.dir_path = dir_path
        self.window_size = window_size
        self.stride = stride

        self.file_list = [f for f in os.listdir(dir_path) if f.endswith(".parquet")]
        self.file_list.sort()  # Ensure files are in order

        self.data = self._load_and_preprocess_data()
        self.valid_indices = self._get_valid_indices()

    def _load_and_preprocess_data(self):
        all_data = []
        for file in self.file_list:
            df = pd.read_parquet(os.path.join(self.dir_path, file))

            # Check if '날짜' column exists, if not, create one from the index
            if "날짜" not in df.columns:
                if isinstance(df.index, pd.DatetimeIndex):
                    df["날짜"] = df.index
                else:
                    print(
                        f"Warning: '날짜' column not found in {file} and index is not datetime. Using a range index instead."
                    )
                    df["날짜"] = pd.date_range(
                        start="2000-01-01", periods=len(df), freq="D"
                    )

            df["날짜"] = pd.to_datetime(df["날짜"])
            df = df.set_index("날짜").sort_index()
            all_data.append(df)

        combined_data = pd.concat(all_data)
        return combined_data

    def _get_valid_indices(self):
        valid_indices = []
        for i in range(0, len(self.data) - self.window_size + 1, self.stride):
            window = self.data.iloc[i : i + self.window_size]
            if not window.isnull().values.any():
                valid_indices.append(i)
        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.window_size

        window_data = self.data.iloc[start_idx:end_idx]

        # Calculate indicators for the window
        indicators = calculate_indicators(window_data)

        # Convert indicators to a numpy array
        indicator_array = np.column_stack(
            [
                indicators["ACF"],
                indicators["Buffett"].reshape(-1, 1),
                indicators["DE"].reshape(-1, 1),
                indicators["DEB"].reshape(-1, 1),
                indicators["FracDiff"].reshape(-1, 1),
                indicators["PivotPoints"],
                indicators["SN"].reshape(-1, 1),
                indicators["STFA"].reshape(-1, 1),
                indicators["STSL"].reshape(-1, 1),
                indicators["TimeDelay"].reshape(-1, 1),
                indicators["CalVIX"].reshape(-1, 1),
                indicators["Williams"].reshape(-1, 1),
            ]
        )

        # Get the target (next day's closing price)
        if end_idx < len(self.data):
            target = (
                self.data.iloc[end_idx]["종가"]
                if "종가" in self.data.columns
                else np.nan
            )
        else:
            target = np.nan

        return {
            "date": window_data.index[-1].strftime("%Y-%m-%d"),  # Convert to string
            "features": window_data.values.astype(np.float32),  # Ensure numpy array
            "indicators": indicator_array.astype(np.float32),  # Ensure numpy array
            "target": np.float32(target),  # Ensure numpy scalar
        }

    def get_feature_dim(self):
        return self.data.shape[1] + sum(
            [
                5,  # ACF
                1,  # Buffett
                1,  # DE
                1,  # DEB
                1,  # FracDiff
                7,  # PivotPoints
                1,  # SN
                1,  # STFA
                1,  # STSL
                1,  # TimeDelay
                1,  # CalVIX
                1,  # Williams
            ]
        )


def calculate_indicators(df):
    # This is a simplified version. In practice, you'd implement each indicator calculation.
    return {
        "ACF": np.random.rand(len(df), 5),
        "Buffett": np.random.rand(len(df)),
        "DE": np.random.rand(len(df)),
        "DEB": np.random.rand(len(df)),
        "FracDiff": np.random.rand(len(df)),
        "PivotPoints": np.random.rand(len(df), 7),
        "SN": np.random.rand(len(df)),
        "STFA": np.random.rand(len(df)),
        "STSL": np.random.rand(len(df)),
        "TimeDelay": np.random.rand(len(df)),
        "CalVIX": np.random.rand(len(df)),
        "Williams": np.random.rand(len(df)),
    }


def main():
    # Set the path to the directory containing your parquet files
    data_dir = r"D:\Workspace\DnS\data"

    # Initialize the SlidingWindowDataLoader
    dataset = SlidingWindowDataLoader(data_dir, window_size=5, stride=2)

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"Total number of samples: {len(dataset)}")
    print(f"Feature dimension: {dataset.get_feature_dim()}")

    # Print some samples
    for i, sample in enumerate(dataloader):
        print(f"\nSample {i+1}:")
        print(f"Date: {sample['date'][0]}")
        print(f"Features shape: {sample['features'].shape}")
        print(f"Indicators shape: {sample['indicators'].shape}")
        print(f"Target: {sample['target'].item()}")

        if i == 4:  # Print only first 5 samples
            break


if __name__ == "__main__":
    main()
