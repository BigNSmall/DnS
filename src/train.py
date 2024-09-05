import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import joblib

# Indicator calculation functions
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
        self.indicators = calculate_indicators(df, window_size)
        self.valid_indices = self._get_valid_indices()
        self.scaler = MinMaxScaler()
        self._fit_scaler()

    def _get_valid_indices(self):
        valid_indices = []
        for i in range(0, len(self.df) - self.window_size + 1, self.stride):
            window = self.df.iloc[i : i + self.window_size]
            if not window.isnull().values.any() and not any(
                np.isnan(indicator[i : i + self.window_size]).any()
                for indicator in self.indicators.values()
            ):
                valid_indices.append(i)
        return valid_indices

    def _fit_scaler(self):
        all_data = []
        for i in self.valid_indices:
            window_data = self._get_window_data(i)
            all_data.append(window_data)
        all_data = np.concatenate(all_data, axis=0)
        self.scaler.fit(all_data)

    def _get_window_data(self, start_idx):
        end_idx = start_idx + self.window_size
        window_data = np.concatenate(
            [
                v[start_idx:end_idx].reshape(self.window_size, -1)
                for v in self.indicators.values()
            ],
            axis=1,
        )
        return window_data

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        window_data = self._get_window_data(start_idx)
        scaled_data = self.scaler.transform(window_data)
        return torch.FloatTensor(scaled_data)

    def get_dates(self, idx):
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.window_size
        return self.df.index[start_idx:end_idx]


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
            inputs = batch.to(device)
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
                inputs = batch.to(device)
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


def save_model(model, scaler, model_path, scaler_path):
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)


def load_model(model, model_path, scaler_path, device):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    scaler = joblib.load(scaler_path)
    return model, scaler


def plot_losses(train_losses, test_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Autoencoder Training Progress")
    plt.show()


def visualize_reconstruction(model, test_loader, device, dataset):
    model.eval()
    with torch.no_grad():
        # Get a batch of test data
        batch = next(iter(test_loader))
        inputs = batch.to(device)  # Ensure inputs are on the correct device
        original_shape = inputs.shape
        inputs_flattened = inputs.view(inputs.size(0), -1)

        # Get model reconstruction
        outputs = model(inputs_flattened)

        # Convert to numpy for plotting (move back to CPU)
        inputs = inputs.cpu().numpy()
        outputs = outputs.cpu().numpy().reshape(original_shape)

        # Plot all indicators in one graph
        fig, ax = plt.subplots(figsize=(20, 10))

        # Define color palette
        num_indicators = inputs.shape[2]
        colors = plt.colormaps["tab20"](np.linspace(0, 1, num_indicators * 2))

        indicator_names = list(dataset.indicators.keys())

        for i in range(num_indicators):
            if i < len(indicator_names):
                indicator_name = indicator_names[i]
            else:
                indicator_name = f"Indicator {i+1}"

            # Plot original
            ax.plot(
                inputs[0, :, i],
                color=colors[i * 2],
                label=f"{indicator_name} (Original)",
            )
            # Plot reconstructed
            ax.plot(
                outputs[0, :, i],
                color=colors[i * 2 + 1],
                linestyle="--",
                label=f"{indicator_name} (Reconstructed)",
            )

        ax.set_title("Original vs Reconstructed Indicators", fontsize=16)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Normalized Value")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Load data
    df = pd.read_parquet(r"D:\Workspace\DnS\data\삼성전자_20190825_20240825.parquet")

    # Create SlidingWindowDataset instance
    dataset = SlidingWindowDataset(df, window_size=5, stride=3)

    # Train-test split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoader
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch.view(sample_batch.size(0), -1).size(1)
    model = Autoencoder(input_dim)

    # Set training parameters
    num_epochs = 1000
    learning_rate = 0.0001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Train model
    train_losses, test_losses = train_autoencoder(
        model, train_loader, test_loader, num_epochs, learning_rate, device
    )

    print("Training completed!")

    # Save model and scaler
    save_model(model, dataset.scaler, "autoencoder_model.pth", "minmax_scaler.pkl")

    # Plot training results
    plot_losses(train_losses, test_losses)

    # Visualize reconstruction
    visualize_reconstruction(model, test_loader, device, dataset)

    # Example of loading the model and scaler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model = Autoencoder(input_dim).to(device)
    loaded_model, loaded_scaler = load_model(
        loaded_model, "autoencoder_model.pth", "minmax_scaler.pkl", device
    )
    print(f"Model and scaler loaded successfully on {device}!")

    # Visualize reconstruction with loaded model
    visualize_reconstruction(loaded_model, test_loader, device, dataset)
