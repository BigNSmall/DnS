import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import umap
from torch.utils.data import DataLoader
from typing import List, Tuple

# Assuming SlidingWindowDataLoader and Autoencoder are imported from the previous file
from autoencoder_training import SlidingWindowDataLoader, Autoencoder


def load_model(model_path: str, input_dim: int, device: torch.device) -> Autoencoder:
    model = Autoencoder(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def get_embeddings(
    model: Autoencoder, data_loader: DataLoader, device: torch.device
) -> np.ndarray:
    embeddings = []
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch["window"].to(device)
            embedding = model.encode(inputs)
            embeddings.append(embedding.cpu().numpy())
    return np.vstack(embeddings)


def perform_clustering(
    embeddings: np.ndarray, n_clusters: int
) -> Tuple[np.ndarray, KMeans]:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    return cluster_labels, kmeans


def visualize_clusters(
    embeddings: np.ndarray, cluster_labels: np.ndarray, n_clusters: int
):
    reducer = umap.UMAP(random_state=42)
    scaled_embeddings = StandardScaler().fit_transform(embeddings)
    embedding_2d = reducer.fit_transform(scaled_embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=cluster_labels,
        cmap="viridis",
        alpha=0.7,
    )
    plt.colorbar(scatter)
    plt.title(f"UMAP visualization of {n_clusters} clusters")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.show()


def main():
    # Parameters
    data_dir = "./data"
    model_path = "autoencoder_model.pth"
    n_clusters = 5  # You can adjust this
    window_size = 5
    stride = 3
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    dataset = SlidingWindowDataLoader(
        dir=data_dir, window_size=window_size, stride=stride
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load model
    sample = next(iter(data_loader))
    input_dim = sample["window"].shape[1]
    model = load_model(model_path, input_dim, device)

    # Get embeddings
    embeddings = get_embeddings(model, data_loader, device)

    # Perform clustering
    cluster_labels, kmeans = perform_clustering(embeddings, n_clusters)

    # Visualize clusters
    visualize_clusters(embeddings, cluster_labels, n_clusters)

    print(f"Clustering completed. {n_clusters} clusters formed.")
    print(f"Cluster sizes: {np.bincount(cluster_labels)}")


if __name__ == "__main__":
    main()
