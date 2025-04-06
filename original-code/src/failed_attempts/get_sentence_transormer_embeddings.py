import pickle

from pathlib import Path
from sentence_transformers import SentenceTransformer

from src.base_data import ClientData, SPLIT_DATA_SPLIT1_DIR, load_split_data


TRAIN_SENTENCE_EMBEDDINGS_FILE = SPLIT_DATA_SPLIT1_DIR/'train_embeddings.pkl'
VAL_SENTENCE_EMBEDDINGS_FILE = SPLIT_DATA_SPLIT1_DIR/'val_embeddings.pkl'
TEST_SENTENCE_EMBEDDINGS_FILE = SPLIT_DATA_SPLIT1_DIR/'test_embeddings.pkl'


def dict_to_text(d: dict, section_name: str = "") -> str:
    """Convert nested dictionary to a readable string format."""
    lines = [f"{section_name}:" if section_name else ""]
    def recurse(k, v, prefix=""):
        if isinstance(v, dict):
            for subk, subv in v.items():
                recurse(subk, subv, f"{prefix}{k} -> ")
        elif isinstance(v, list):
            for i, item in enumerate(v):
                recurse(f"{k}[{i}]", item, prefix)
        else:
            lines.append(f"{prefix}{k}: {v}")
    for key, value in d.items():
        recurse(key, value)
    return "\n".join(lines)


def clientdata_to_text(client: ClientData) -> str:
    """Convert ClientData object to a single string."""
    sections = [
        dict_to_text(client.account_form, "Account Form"),
        dict_to_text(client.client_description, "Client Description"),
        dict_to_text(client.client_profile, "Client Profile"),
        dict_to_text(client.passport, "Passport")
    ]
    return "\n\n".join(sections)


def get_embeddings(
    clients: list[ClientData],
    savefile: Path,
    model_name: str = 'sentence-transformers/all-mpnet-base-v2'
):
    # load the model
    model = SentenceTransformer(model_name).to('cuda')
    # get embeddings
    texts = [clientdata_to_text(client) for client in clients]
    embeddings = model.encode(texts, batch_size=8, show_progress_bar=True)

    # convert to numpy array and save
    embeddings = embeddings
    with open(savefile, 'wb') as f:
        pickle.dump(embeddings, f)
    return embeddings


if __name__ == "__main__":
    train_data, val_data, test_data = load_split_data(SPLIT_DATA_SPLIT1_DIR)

    # save embeddings
    get_embeddings(val_data, VAL_SENTENCE_EMBEDDINGS_FILE)
    get_embeddings(test_data, TEST_SENTENCE_EMBEDDINGS_FILE)
    get_embeddings(train_data, TRAIN_SENTENCE_EMBEDDINGS_FILE)
