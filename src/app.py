import streamlit as st
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from train import (
    SlidingWindowDataset,
    Autoencoder,
    train_autoencoder,
    save_model,
    load_model,
    visualize_reconstruction,
)
from cluster import encode_data, perform_clustering, visualize_clusters


# Streamlit app
def main():
    st.set_page_config(
        page_title="Autoencoder Training and Clustering App", layout="wide"
    )
    st.title("Autoencoder Training and Clustering App")

    # Sidebar for navigation
    page = st.sidebar.selectbox(
        "Choose a page", ["Train Autoencoder", "Perform Clustering"]
    )

    if page == "Train Autoencoder":
        train_page()
    elif page == "Perform Clustering":
        clustering_page()


def train_page():
    st.header("Train Autoencoder")

    # File uploader for parquet data
    uploaded_file = st.file_uploader("Choose a parquet file", type="parquet")
    if uploaded_file is not None:
        df = pd.read_parquet(uploaded_file)
        st.write("Data loaded successfully!")

        # Model parameters
        window_size = st.slider("Window Size", min_value=1, max_value=10, value=5)
        stride = st.slider("Stride", min_value=1, max_value=5, value=3)
        num_epochs = st.slider(
            "Number of Epochs", min_value=10, max_value=1000, value=100
        )
        learning_rate = st.number_input(
            "Learning Rate", min_value=0.0001, max_value=0.1, value=0.001, format="%.4f"
        )

        if st.button("Train Model"):
            with st.spinner("Training in progress..."):
                # Create dataset
                dataset = SlidingWindowDataset(
                    df, window_size=window_size, stride=stride
                )

                # Train-test split
                train_size = int(0.8 * len(dataset))
                test_size = len(dataset) - train_size
                train_dataset, test_dataset = train_test_split(
                    dataset, test_size=0.2, random_state=42
                )

                # Create DataLoader
                batch_size = 64
                train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True
                )
                test_loader = torch.utils.data.DataLoader(
                    test_dataset, batch_size=batch_size, shuffle=False
                )

                # Initialize model
                sample_batch = next(iter(train_loader))
                input_dim = sample_batch.view(sample_batch.size(0), -1).size(1)
                model = Autoencoder(input_dim)

                # Set device
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)

                # Train model
                train_losses, test_losses = train_autoencoder(
                    model, train_loader, test_loader, num_epochs, learning_rate, device
                )

                # Save model and scaler
                save_model(
                    model, dataset.scaler, "autoencoder_model.pth", "minmax_scaler.pkl"
                )

                st.success("Training completed!")

                # Plot training results
                fig, ax = plt.subplots()
                ax.plot(train_losses, label="Train Loss")
                ax.plot(test_losses, label="Test Loss")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.legend()
                ax.set_title("Autoencoder Training Progress")
                st.pyplot(fig)

                # Visualize reconstruction
                fig = visualize_reconstruction(model, test_loader, device, dataset)
                st.pyplot(fig)


def clustering_page():
    st.header("Perform Clustering")

    # File uploader for parquet data
    uploaded_file = st.file_uploader("Choose a parquet file", type="parquet")
    if uploaded_file is not None:
        df = pd.read_parquet(uploaded_file)
        st.write("Data loaded successfully!")

        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = SlidingWindowDataset(df, window_size=5, stride=3)
        sample_batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1)))
        input_dim = sample_batch.view(sample_batch.size(0), -1).size(1)
        model = Autoencoder(input_dim).to(device)
        model, _ = load_model(
            model, "autoencoder_model.pth", "minmax_scaler.pkl", device
        )
        st.write(f"Model loaded successfully on {device}!")

        # Encode data
        encoded_data = encode_data(model, dataset, device)
        st.write(f"Encoded data shape: {encoded_data.shape}")

        # Select clustering algorithms
        clustering_options = [
            "K-means",
            "Hierarchical",
            "DBSCAN",
            "OPTICS",
            "Spectral",
            "GaussianMixture",
            "MeanShift",
        ]
        selected_algorithms = st.multiselect(
            "Select clustering algorithms", clustering_options
        )

        if st.button("Perform Clustering"):
            with st.spinner("Clustering in progress..."):
                # Perform clustering
                labels_dict = perform_clustering(encoded_data)
                selected_labels = {
                    k: v for k, v in labels_dict.items() if k in selected_algorithms
                }

                # Visualize clusters
                fig = visualize_clusters(encoded_data, selected_labels)
                st.pyplot(fig)


if __name__ == "__main__":
    main()
