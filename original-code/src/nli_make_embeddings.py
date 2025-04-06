from src.nli_utils import get_model_and_tokenizer, roberta_embeddings
from src.base_data import ClientData, SPLIT_DATA_SPLIT1_DIR, load_split_data

train_data, val_data, test_data = load_split_data(SPLIT_DATA_SPLIT1_DIR)

TRAIN_SENTENCE_EMBEDDINGS_FILE = SPLIT_DATA_SPLIT1_DIR/'train_embeddings.pkl'
VAL_SENTENCE_EMBEDDINGS_FILE = SPLIT_DATA_SPLIT1_DIR/'val_embeddings.pkl'
TEST_SENTENCE_EMBEDDINGS_FILE = SPLIT_DATA_SPLIT1_DIR/'test_embeddings.pkl'


tok, model = get_model_and_tokenizer()
embs = roberta_embeddings(val_data, tok, model, VAL_SENTENCE_EMBEDDINGS_FILE)
embs = roberta_embeddings(test_data, tok, model, TEST_SENTENCE_EMBEDDINGS_FILE)
embs = roberta_embeddings(train_data, tok, model, TRAIN_SENTENCE_EMBEDDINGS_FILE)