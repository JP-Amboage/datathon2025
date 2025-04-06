import torch
import pickle
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

from src.rule_based_models import SimpleModel
from src.models.nli_model import ProfileAttentionClassifier
from src.base_data import ClientData, SPLIT_DATA_SPLIT1_DIR, load_split_data

TRAIN_SENTENCE_EMBEDDINGS_FILE = SPLIT_DATA_SPLIT1_DIR/'train_embeddings.pkl'
VAL_SENTENCE_EMBEDDINGS_FILE = SPLIT_DATA_SPLIT1_DIR/'val_embeddings.pkl'
TEST_SENTENCE_EMBEDDINGS_FILE = SPLIT_DATA_SPLIT1_DIR/'test_embeddings.pkl'

MODEL_SAVEPATH = SPLIT_DATA_SPLIT1_DIR/'nli_model.pt'
MODEL_CHECKPOINT = SPLIT_DATA_SPLIT1_DIR/'nli_model_checkpoint.pt'



def filter_data(client_data: list[ClientData], embeddings: np.ndarray, rule_based_model: SimpleModel):
    """Filter the data based on the rule-based model predictions."""
    filtered_data = []
    filtered_embeddings = []
    for i, cd in enumerate(tqdm(client_data, desc="Filtering data")):
        pred = rule_based_model.predict(cd)
        if pred == 0:
            filtered_data.append(cd)
            filtered_embeddings.append(embeddings[i])
    print(f'Kept {len(filtered_data)} out of {len(client_data)} data points.')
    return filtered_data, filtered_embeddings


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, client_data: list[ClientData]):
        self.embeddings = embeddings
        self.client_data = client_data

    def __len__(self):
        return len(self.client_data)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.client_data[idx].label


def get_class_weights(client_data: list[ClientData]):
    """Get the class weights for the dataset based on the distribution of labels."""
    class_weights = [0, 0]
    total_accept, total_reject = 0, 0
    for cd in client_data:
        if cd.label == 0:
            total_accept += 1
        else:
            total_reject += 1
    print(f"Total accept: {total_accept}, Total reject: {total_reject}")
    total = total_accept + total_reject
    class_weights[0] = total / (2 * total_accept)
    class_weights[1] = total / (2 * total_reject)
    print(f"Class weights: {class_weights}")
    return torch.tensor(class_weights, dtype=torch.float32)



def main():
    train_data, val_data, test_data = load_split_data(SPLIT_DATA_SPLIT1_DIR)
    with open(TRAIN_SENTENCE_EMBEDDINGS_FILE, 'rb') as f:
        train_embeddings = pickle.load(f)
    with open(VAL_SENTENCE_EMBEDDINGS_FILE, 'rb') as f:
        val_embeddings = pickle.load(f)
    with open(TEST_SENTENCE_EMBEDDINGS_FILE, 'rb') as f:
        test_embeddings = pickle.load(f)

    rule_based_model = SimpleModel()

    filtered_train_data, filtered_train_embeddings = filter_data(
        train_data, train_embeddings, rule_based_model
    )
    filtered_val_data, filtered_val_embeddings = filter_data(
        val_data, val_embeddings, rule_based_model
    )
    filtered_test_data, filtered_test_embeddings = filter_data(
        test_data, test_embeddings, rule_based_model
    )

    train_ds = EmbeddingDataset(
        filtered_train_embeddings, filtered_train_data
    )
    val_ds = EmbeddingDataset(
        filtered_val_embeddings, filtered_val_data
    )
    test_ds = EmbeddingDataset(
        filtered_test_embeddings, filtered_test_data
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=32,
        shuffle=True,
        # num_workers=4,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=32,
        shuffle=False,
        # num_workers=4,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=32,
        shuffle=False,
        # num_workers=4,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ProfileAttentionClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)

    # cls_weight = get_class_weights(filtered_train_data)
    # pos_weight = torch.Tensor([cls_weight[1]]).to(device)
    # criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)

    criterion = torch.nn.BCEWithLogitsLoss().to(device)

    best_val_acc = 0.0

    # epochs = 100
    # for epoch in range(epochs):
    #     total_train_loss = 0.0
    #     model.train()
    #     for i, (embeddings, labels) in enumerate(train_loader):
    #         embeddings, labels = embeddings.to(device), labels.to(device).float()

    #         optimizer.zero_grad()
    #         outputs = model(embeddings)
    #         loss = criterion(outputs.squeeze(), labels)
    #         loss.backward()
    #         optimizer.step()
    #         total_train_loss += loss.item()
    #         if i % 50 == 0:
    #             print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")

    #     # Validation step
    #     total_val_loss, total_val_acc, total_val_f1 = 0.0, 0.0, 0.0
    #     model.eval()
    #     with torch.no_grad():
    #         for batch in val_loader:
    #             embeddings, labels = batch
    #             embeddings = embeddings.to(device)
    #             labels = labels.to(device).float()

    #             outputs = model(embeddings).squeeze(dim=-1)
    #             loss = criterion(outputs.squeeze(), labels)
    #             total_val_loss += loss.item()

    #             preds = (torch.sigmoid(outputs) >= 0.5).float()
    #             total_val_acc += accuracy_score(labels.cpu(), preds.cpu())
    #             total_val_f1 += f1_score(labels.cpu(), preds.cpu(), average='macro')
    #     val_loss = total_val_loss / len(val_loader)
    #     val_acc = total_val_acc / len(val_loader)
    #     val_f1 = total_val_f1 / len(val_loader)
    #     print(f"Epoch [{epoch+1}/{epochs}], Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

    #     # checkpoint if accuracy improved
    #     if val_acc > best_val_acc:
    #         best_val_acc = val_acc
    #         torch.save(model.state_dict(), MODEL_CHECKPOINT)
    #         print(f"Model checkpoint saved at epoch {epoch+1}")

    fp_scores = []
    fn_scores = []

    # get val_acc
    print(f'Best validation accuracy: {best_val_acc:.4f}')
    correct_predicted = len(val_data) - len(filtered_val_data)
    with torch.no_grad():
        model.load_state_dict(torch.load(MODEL_CHECKPOINT))
        model.eval()
        print("Evaluating on validation set...")
        for i, (embeddings, labels) in enumerate(val_loader):
            embeddings, labels = embeddings.to(device), labels.to(device).float()
            outputs = model(embeddings).squeeze(dim=-1)
            preds = (torch.sigmoid(outputs) >= 0.7).float()
            num_correct = (labels == preds).sum().item()
            correct_predicted += num_correct
            print(f"Val Batch {i+1}/{len(val_loader)}, Correct: {num_correct}")

            fp_indices = (preds == 1) & (labels == 0)
            fn_indices = (preds == 0) & (labels == 1)
            fp_scores.extend(torch.sigmoid(outputs[fp_indices]).cpu().numpy())
            fn_scores.extend(torch.sigmoid(outputs[fn_indices]).cpu().numpy())

    val_acc = correct_predicted / len(val_data)
    print(f"Combined Validation accuracy: {val_acc:.4f}")

    # scatter plot of fp_scores and fn_scores
    import matplotlib.pyplot as plt
    plt.scatter(fp_scores, [0] * len(fp_scores), color='red', label='False Positives')
    plt.scatter(fn_scores, [1] * len(fn_scores), color='blue', label='False Negatives')
    plt.xlabel('Model Score')
    plt.ylabel('Label')
    plt.title('False Positives and False Negatives')
    plt.legend()
    plt.savefig(SPLIT_DATA_SPLIT1_DIR/'fp_fn_scatter.png')


if __name__ == "__main__":
    main()
