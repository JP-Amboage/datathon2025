import torch
import pickle
import torch.nn as nn

from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader

from src.base_data import ClientData, SPLIT_DATA_SPLIT1_DIR, load_split_data
from src.get_sentence_transormer_embeddings import (
    TRAIN_SENTENCE_EMBEDDINGS_FILE,
    VAL_SENTENCE_EMBEDDINGS_FILE,
    TEST_SENTENCE_EMBEDDINGS_FILE,
)


class EmbeddingDataset(Dataset):
    """A dataset that takes as input embeddings and outputs
        a binary value."""

    def __init__(self, client_data: list[ClientData], embeddings: torch.Tensor):
        self.client_data = client_data
        self.embeddings = embeddings

    def __len__(self):
        return len(self.client_data)

    def __getitem__(self, idx):
        return self.embeddings[idx, :], self.client_data[idx].label


class Model(nn.Module):
    """A model that takes as input embeddings and outputs
        a binary value."""

    def __init__(self, embedding_dim: int = 768):
        super(Model, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        x = self.head(x)
        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the output of the model."""
        with torch.no_grad():
            x = self.forward(x)
            x = torch.sigmoid(x)
            x = torch.round(x)
        return x


def main():
    train_data, val_data, test_data = load_split_data(SPLIT_DATA_SPLIT1_DIR)

    # load the embeddings
    with open(TRAIN_SENTENCE_EMBEDDINGS_FILE, 'rb') as f:
        train_embeddings = pickle.load(f)
    with open(VAL_SENTENCE_EMBEDDINGS_FILE, 'rb') as f:
        val_embeddings = pickle.load(f)
    with open(TEST_SENTENCE_EMBEDDINGS_FILE, 'rb') as f:
        test_embeddings = pickle.load(f)

    # make the datasets
    train_dataset = EmbeddingDataset(train_data, train_embeddings)
    val_dataset = EmbeddingDataset(val_data, val_embeddings)
    test_dataset = EmbeddingDataset(test_data, test_embeddings)

    # make the dataloaders
    train_dl = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_dl = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # make the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(embedding_dim=train_embeddings.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # training loop
    for epoch in range(300):
        model.train()
        total_loss = 0.0
        for x, y in train_dl:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device).float()
            y_hat = model(x)
            loss = criterion(y_hat.squeeze(), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dl):.4f}")

        model.eval()
        total_val_loss = 0.0
        total_acc = 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device).float()
                y_hat = model(x)
                loss = criterion(y_hat.squeeze(), y)
                total_val_loss += loss.item()
                y_pred = (y_hat >= 0.5).float()
                acc = accuracy_score(y.cpu(), y_pred.cpu())
                total_acc += acc
        avg_val_loss = total_val_loss / len(val_dl)
        avg_acc = total_acc / len(val_dl)
        print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {avg_acc:.4f}\n")


if __name__ == "__main__":
    main()
