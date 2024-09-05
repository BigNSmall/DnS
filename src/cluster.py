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

from train import SlidingWindowDataset

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
        "2D PCA": lambda data: PCA(n_components=2, random_state=42).fit_transform(data),
        "2D UMAP": lambda data: umap.UMAP(
            n_components=2, random_state=42
        ).fit_transform(data),
    }

    n_rows = len(reduction_methods)
    n_cols = len(labels_dict)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False
    )
    fig.suptitle(
        "Clustering Results with Different Dimension Reduction Methods", fontsize=16
    )

    for i, (reduction_name, reduction_func) in enumerate(reduction_methods.items()):
        reduced_data = reduction_func(encoded_data)

        for j, (cluster_name, labels) in enumerate(labels_dict.items()):
            ax = axes[i, j]
            scatter = ax.scatter(
                reduced_data[:, 0],
                reduced_data[:, 1],
                c=labels,
                cmap="viridis",
                alpha=0.7,
            )
            ax.set_title(f"{reduction_name}\n{cluster_name}")
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(scatter, ax=ax, aspect=40, pad=0.01)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


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
