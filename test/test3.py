import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    DBSCAN,
    OPTICS,
    SpectralClustering,
    MeanShift,
)
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm
import warnings

# Indicator calculation functions
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

# 1. Data Preparation and Model Definition


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


# 2. Load Trained Model and Data


def load_model(model_path, scaler_path, device):
    # Load the state dict
    state_dict = torch.load(model_path, map_location=device)

    # Get the input dimension from the first layer of the encoder
    input_dim = state_dict["encoder.0.weight"].shape[1]

    # Create a new model with the correct input dimension
    model = Autoencoder(input_dim).to(device)

    # Load the state dict into the new model
    model.load_state_dict(state_dict)

    scaler = joblib.load(scaler_path)
    return model, scaler


# 3. Encoding Data


def encode_data(model, dataset, device):
    model.eval()
    encoded_data = []

    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding data"):
            inputs = batch.to(device)
            inputs = inputs.view(inputs.size(0), -1)  # Flatten the input
            encoded = model.encode(inputs)
            encoded_data.append(encoded.cpu().numpy())

    return np.vstack(encoded_data)


# 4. Clustering


def perform_clustering(encoded_data):
    # K-means clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans_labels = kmeans.fit_predict(encoded_data)

    # Hierarchical clustering
    hierarchical = AgglomerativeClustering(n_clusters=5)
    hierarchical_labels = hierarchical.fit_predict(encoded_data)

    # DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(encoded_data)

    return kmeans_labels, hierarchical_labels, dbscan_labels


# 5. Visualization
def perform_clustering(encoded_data):
    n_clusters = 5  # 클러스터 수를 5로 가정

    clustering_methods = {
        "K-means": KMeans(n_clusters=n_clusters, random_state=42),
        "Hierarchical": AgglomerativeClustering(n_clusters=n_clusters),
        "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
        "OPTICS": OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.05),
        "Spectral": SpectralClustering(n_clusters=n_clusters, random_state=42),
        "GaussianMixture": GaussianMixture(n_components=n_clusters, random_state=42),
        "MeanShift": MeanShift(bandwidth=2),
    }

    labels_dict = {}
    for name, method in clustering_methods.items():
        print(f"Performing {name} clustering...")
        labels = method.fit_predict(encoded_data)
        labels_dict[name] = labels

    return labels_dict


# 5. Visualization (수정됨)
def visualize_clusters(encoded_data, labels_dict):
    reduction_methods = {
        "2D t-SNE": lambda data: TSNE(n_components=2, random_state=42).fit_transform(
            data
        ),
        "3D PCA": lambda data: PCA(n_components=3, random_state=42).fit_transform(data),
        "2D UMAP": lambda data: umap.UMAP(n_components=2, n_jobs=-1).fit_transform(
            data
        ),
    }

    n_rows = len(reduction_methods)
    n_cols = len(labels_dict)
    fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))

    plt.rcParams["font.family"] = "DejaVu Sans"

    for i, (reduction_name, reduction_func) in enumerate(reduction_methods.items()):
        reduced_data = reduction_func(encoded_data)

        for j, (cluster_name, labels) in enumerate(labels_dict.items()):
            ax = fig.add_subplot(
                n_rows,
                n_cols,
                i * n_cols + j + 1,
                projection="3d" if reduction_name == "3D PCA" else None,
            )

            if reduction_name == "3D PCA":
                scatter = ax.scatter(
                    reduced_data[:, 0],
                    reduced_data[:, 1],
                    reduced_data[:, 2],
                    c=labels,
                    cmap="viridis",
                )
                ax.set_xlabel("Component 1")
                ax.set_ylabel("Component 2")
                ax.set_zlabel("Component 3")
            else:
                scatter = ax.scatter(
                    reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap="viridis"
                )
                ax.set_xlabel("Component 1")
                ax.set_ylabel("Component 2")

            ax.set_title(f"{reduction_name} - {cluster_name}")
            plt.colorbar(scatter, ax=ax)

    plt.tight_layout()
    plt.show()


# Main execution (수정됨)
if __name__ == "__main__":
    # Load data
    df = pd.read_parquet(r"D:\Workspace\DnS\data\삼성전자_20190825_20240825.parquet")

    # Create SlidingWindowDataset instance
    dataset = SlidingWindowDataset(df, window_size=5, stride=3)

    # Load trained model and scaler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model("autoencoder_model.pth", "minmax_scaler.pkl", device)
    print(f"Model loaded successfully on {device}!")

    # Encode data
    encoded_data = encode_data(model, dataset, device)
    print(f"Encoded data shape: {encoded_data.shape}")

    # Perform clustering with all methods
    labels_dict = perform_clustering(encoded_data)

    # Visualize clusters
    visualize_clusters(encoded_data, labels_dict)

    print("Clustering and visualization completed!")
