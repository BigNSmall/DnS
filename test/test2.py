import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import platform


# 한글 폰트 설정
def set_korean_font():
    if platform.system() == "Windows":
        font_name = font_manager.FontProperties(
            fname="c:/Windows/Fonts/malgun.ttf"
        ).get_name()
        rc("font", family=font_name)
    elif platform.system() == "Darwin":  # macOS
        rc("font", family="AppleGothic")
    else:  # Linux
        rc("font", family="NanumGothic")
    plt.rcParams["axes.unicode_minus"] = False


# LazyDataLoader 클래스
class LazyDataLoader:
    def __init__(self, file_paths: List[str]):
        self.file_paths = file_paths
        self.data_cache = {}

    def __getitem__(self, index: int) -> pd.DataFrame:
        if index not in self.data_cache:
            file_path = self.file_paths[index]
            self.data_cache[index] = pd.read_parquet(file_path)
        return self.data_cache[index]

    def __len__(self) -> int:
        return len(self.file_paths)


# SlidingWindowDataset 클래스
class SlidingWindowDataset(Dataset):
    def __init__(
        self, lazy_loader: LazyDataLoader, window_size: int = 5, stride: int = 2
    ):
        self.lazy_loader = lazy_loader
        self.window_size = window_size
        self.stride = stride
        self.file_indices = []
        self.data_indices = []
        self.scalers = {}
        self.feature_names = None

        self._preprocess_data()

    def _preprocess_data(self):
        for file_idx in range(len(self.lazy_loader)):
            df = self.lazy_loader[file_idx]
            if self.feature_names is None:
                self.feature_names = df.columns.tolist()
            valid_indices = self._get_valid_indices(df)
            self.file_indices.extend([file_idx] * len(valid_indices))
            self.data_indices.extend(valid_indices)

            scaler = MinMaxScaler()
            scaler.fit(df)
            self.scalers[file_idx] = scaler

    def _get_valid_indices(self, df: pd.DataFrame) -> List[int]:
        return [
            i
            for i in range(0, len(df) - self.window_size + 1, self.stride)
            if not np.isnan(df.iloc[i : i + self.window_size].values).any()
        ]

    def __len__(self) -> int:
        return len(self.data_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_idx = self.file_indices[idx]
        start_idx = self.data_indices[idx]
        end_idx = start_idx + self.window_size

        df = self.lazy_loader[file_idx]
        window = df.iloc[start_idx:end_idx]

        normalized_window = self.scalers[file_idx].transform(window)
        normalized_window = pd.DataFrame(normalized_window, columns=self.feature_names)

        return {"window": torch.FloatTensor(normalized_window.values)}

    def inverse_transform(
        self, normalized_data: np.ndarray, file_idx: int
    ) -> pd.DataFrame:
        inverse_data = self.scalers[file_idx].inverse_transform(normalized_data)
        return pd.DataFrame(inverse_data, columns=self.feature_names)


# Autoencoder 모델
class Autoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64, 32],
        latent_dim: int = 16,
        dropout_rate: float = 0.2,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        attn_output, _ = self.attention(
            encoded.unsqueeze(0), encoded.unsqueeze(0), encoded.unsqueeze(0)
        )
        attn_output = attn_output.squeeze(0)
        encoded = encoded + attn_output
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


# 학습 함수
def train_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
) -> Tuple[List[float], List[float]]:
    criterion = nn.HuberLoss()
    optimizer = optim.NAdam(model.parameters(), lr=learning_rate)

    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = batch["window"].float().to(device)
            inputs = inputs.view(inputs.size(0), -1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

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


# 결과 시각화 함수
def visualize_results(train_losses: List[float], test_losses: List[float]):
    set_korean_font()

    train_losses = np.array(train_losses)
    test_losses = np.array(test_losses)

    if np.isnan(train_losses).any() or np.isnan(test_losses).any():
        print("경고: 손실 값에 NaN이 포함되어 있습니다.")
        train_losses = np.nan_to_num(train_losses)
        test_losses = np.nan_to_num(test_losses)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="훈련 손실", marker="o")
    plt.plot(test_losses, label="테스트 손실", marker="s")
    plt.xlabel("에포크")
    plt.ylabel("손실")
    plt.title("훈련 및 테스트 손실")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.ylim(bottom=0)

    try:
        plt.show()
    except Exception as e:
        print(f"플롯 표시 중 오류 발생: {e}")
        print("대신 이미지로 저장합니다.")
        plt.savefig("training_test_losses.png")

    plt.close()

    print(f"최종 훈련 손실: {train_losses[-1]:.4f}")
    print(f"최종 테스트 손실: {test_losses[-1]:.4f}")
    print(f"최소 훈련 손실: {np.min(train_losses):.4f}")
    print(f"최소 테스트 손실: {np.min(test_losses):.4f}")


# 메인 함수
def main():
    # 데이터 디렉토리 설정
    data_directory = r"D:\Workspace\DnS\data"

    # 파라미터 설정
    window_size = 5
    stride = 3
    batch_size = 64
    num_epochs = 100
    learning_rate = 0.0001
    test_size = 0.2

    # 파일 목록 가져오기
    file_paths = [
        os.path.join(data_directory, f)
        for f in os.listdir(data_directory)
        if f.endswith(".parquet")
    ]

    # Lazy 데이터 로더 생성
    lazy_loader = LazyDataLoader(file_paths)

    # 전체 데이터셋 생성
    full_dataset = SlidingWindowDataset(
        lazy_loader, window_size=window_size, stride=stride
    )

    # 학습 및 테스트 세트 분할
    train_indices, test_indices = train_test_split(
        range(len(full_dataset)), test_size=test_size, shuffle=True
    )
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 모델 초기화
    input_dim = window_size * lazy_loader[0].shape[1]
    model = Autoencoder(input_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 모델 학습
    train_losses, test_losses = train_autoencoder(
        model, train_loader, test_loader, num_epochs, learning_rate, device
    )

    # 결과 시각화
    visualize_results(train_losses, test_losses)

    print("학습 및 시각화가 완료되었습니다.")


if __name__ == "__main__":
    main()
