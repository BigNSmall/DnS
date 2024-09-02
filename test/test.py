import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tqdm import tqdm


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


class Autoencoder(nn.Module):
    def __init__(
        self, input_dim, hidden_dims=[256, 128, 64, 32], latent_dim=16, dropout_rate=0.2
    ):
        super(Autoencoder, self).__init__()

        # Encoder
        encoder_layers = []
        in_dim = input_dim
        for dim in hidden_dims:
            encoder_layers.extend(
                [
                    nn.Linear(in_dim, dim),
                    nn.LayerNorm(dim),
                    nn.PReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            in_dim = dim
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        in_dim = latent_dim
        for dim in reversed(hidden_dims):
            decoder_layers.extend(
                [
                    nn.Linear(in_dim, dim),
                    nn.LayerNorm(dim),
                    nn.PReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            in_dim = dim
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # Attention mechanism
        self.attention = nn.MultiheadAttention(latent_dim, num_heads=4)

    def forward(self, x):
        encoded = self.encoder(x)
        # Apply attention
        attn_output, _ = self.attention(
            encoded.unsqueeze(0), encoded.unsqueeze(0), encoded.unsqueeze(0)
        )
        attn_output = attn_output.squeeze(0)
        # Add residual connection
        encoded = encoded + attn_output
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)


def train_autoencoder(
    model, train_loader, test_loader, num_epochs, learning_rate, device
):
    criterion = nn.HuberLoss()
    optimizer = optim.NAdam(model.parameters(), lr=learning_rate)

    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = batch["window"].float().to(device)
            inputs = inputs.view(inputs.size(0), -1)  # Flatten the input

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Evaluate on test set
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["window"].float().to(device)
                inputs = inputs.view(inputs.size(0), -1)
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}"
        )

    return train_losses, test_losses


def visualize_results(train_losses, test_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss")
    plt.legend()
    plt.show()


def visualize_reconstruction(model, test_loader, device, combined_df):
    model.eval()
    with torch.no_grad():
        # Get a batch of test data
        batch = next(iter(test_loader))
        inputs = batch["window"].float().to(device)
        inputs = inputs.view(inputs.size(0), -1)

        # Get model reconstruction
        outputs = model(inputs)

        # Convert to numpy for plotting
        inputs = inputs.cpu().numpy()
        outputs = outputs.cpu().numpy()

        # Reshape inputs and outputs to match the original data shape
        inputs = inputs.reshape(inputs.shape[0], -1, combined_df.shape[1])
        outputs = outputs.reshape(outputs.shape[0], -1, combined_df.shape[1])

        # Plot all indicators in one graph
        num_indicators = combined_df.shape[1]
        fig, ax = plt.subplots(figsize=(20, 10))

        # Define color palette
        colors = plt.colormaps.get_cmap("tab20")(np.linspace(0, 1, num_indicators * 2))

        for i in range(num_indicators):
            # Plot original
            ax.plot(
                inputs[0, :, i],
                color=colors[i * 2],
                label=f"{combined_df.columns[i]} (Original)",
            )

            # Plot reconstructed
            ax.plot(
                outputs[0, :, i],
                color=colors[i * 2 + 1],
                linestyle="--",
                label=f"{combined_df.columns[i]} (Reconstructed)",
            )

        ax.set_title("Original vs Reconstructed Indicators", fontsize=16)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Normalized Value")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        plt.show()


def main():
    # 데이터 로드
    df = pd.read_parquet(r"D:\Workspace\DnS\data\AJ네트웍스_20190825_20240825.parquet")

    # 파라미터 설정
    window_size = 5
    stride = 3
    batch_size = 64
    num_epochs = 100
    learning_rate = 0.0001
    test_size = 0.2

    # 지표 계산 및 결합
    indicators = calculate_indicators(df, window_size)
    combined_df = combine_indicators(indicators, df)

    # 데이터 분할
    train_df, test_df = train_test_split(combined_df, test_size=test_size, shuffle=True)

    # 데이터셋 및 DataLoader 생성
    train_dataset = SlidingWindowDataLoader(
        train_df, window_size=window_size, stride=stride
    )
    test_dataset = SlidingWindowDataLoader(
        test_df, window_size=window_size, stride=stride
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    main()
