import json
import zipfile
from pathlib import Path

from tqdm import tqdm

from base_data import DataPipeline
from rule_based_models import SimpleModel
from base_data import SPLIT_DATA_SPLIT1_DIR, load_split_data
from tqdm import tqdm


def evaluate(model, client_list):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for client in tqdm(client_list[:]):
        pred = model.predict(client)
        label = client.label
        if pred == 0:
            if label == 0:
                tn += 1
            else:
                fn += 1
                print(
                    f"FN Client data: {client.passport['passport_number']} and {client.client_file}"
                )

        else:
            if label == 0:
                fp += 1
                print(f"FP Client data: {client.passport['passport_number']} and {client.client_profile['currency']}")
            else:
                tp += 1

    print(f"tp: {tp}")
    print(f"fp: {fp}")
    print(f"fn: {fn}")
    print(f"tn: {tn}")
    print(f"accuracy: {(tp + tn)/(tp + tn + fp + fn)}")


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent.resolve().absolute() / "data"
    ROOT_DIRS = [
        data_dir / "datathon_part1.zip",
        data_dir / "datathon_part2.zip",
        data_dir / "datathon_part3.zip",
        data_dir / "datathon_part4.zip",
    ]

    data_pipeline = DataPipeline(
        ROOT_DIRS
    )  # unzips (if not already unzipped) and gets data paths
    # print(f"Data directories: {data_pipeline.data_dirs}")  # prints the data directories

    # CV
    train_paths, val_paths, test_paths = data_pipeline.split_data(
        num_splits=1, train_ratio=0.8, stratified=True
    )
    print(
        f"Loaded {len(train_paths)} training paths, {len(val_paths)} validation paths, and {len(test_paths)} test paths."
    )
    train_data, val_data, test_data = data_pipeline.load_data(
        train_paths, val_paths, test_paths
    )
    # train_data, val_data, test_data = load_split_data(SPLIT_DATA_SPLIT1_DIR)

    model = SimpleModel()
    evaluate(model, train_data)
