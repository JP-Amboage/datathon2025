import json
import pickle
import zipfile

from pathlib import Path
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).parent.parent.resolve().absolute()

data_dir = PROJECT_DIR/'data'
ROOT_DIRS = [
    data_dir/'datathon_part1.zip',
    data_dir/'datathon_part2.zip',
    data_dir/'datathon_part3.zip',
    data_dir/'datathon_part4.zip'
]

SPLIT_DATA_DIR = PROJECT_DIR/'split-loaded-data'
SPLIT_DATA_SPLIT1_DIR = SPLIT_DATA_DIR/'split1'
SPLIT_DATA_SPLIT5_DIR = SPLIT_DATA_DIR/'split5'

@dataclass
class ClientData:
    """Client data."""
    client_file: str
    account_form: dict
    client_description: dict
    client_profile: dict
    passport: dict
    label: int | None = None


def load_split_data(data_dir: Path) -> tuple[list[ClientData], list[ClientData], list[ClientData]]:
    with open(data_dir/'train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open(data_dir/'val.pkl', 'rb') as f:
        val_data = pickle.load(f)
    with open(data_dir/'test.pkl', 'rb') as f:
        test_data = pickle.load(f)
    print(f'Loaded {len(train_data)} training paths, {len(val_data)} validation paths, and {len(test_data)} test paths.')
    return train_data, val_data, test_data


class DataPipeline:
    """Interface for loading data (stratified, training and predicting with models."""

    LABEL_MAP = {
        'Accept': 0,
        'accept': 0,
        'Reject': 1,
        'reject': 1
    }

    def __init__(self, root_dirs: list[Path]):
        self.data_dirs = []
        for root_dir in root_dirs:
            self.data_dirs += [*self._unzip_and_get_data_paths(root_dir)]
        self.data_dirs.sort(key=lambda dir_path: int(dir_path.stem.split('_')[-1]))

    @staticmethod
    def _unzip_and_get_data_paths(zip_path: Path, verbose: bool = False):
        assert zip_path.exists() and zip_path.suffix == '.zip', f"Path  {zip_path} must be a .zip file. {zip_path.suffix}"

        # Define outer extraction directory
        outer_dir = zip_path.with_suffix('')

        # Step 1: Unzip the outer zip if not already unzipped
        if not outer_dir.exists():
            if verbose:
                print(f"Unzipping outer zip to {outer_dir}")
            with zipfile.ZipFile(zip_path, 'r') as outer_zip:
                outer_zip.extractall(outer_dir)
        else:
            if verbose:
                print(f"Outer zip already unzipped at {outer_dir}")

        # Step 2: Go through each sub-zip file and unzip if needed
        data_point_dirs = []
        for subzip_path in outer_dir.glob('*.zip'):
            data_point_dir = outer_dir / subzip_path.stem

            if not data_point_dir.exists():
                if verbose:
                    print(f"Unzipping sub-zip {subzip_path.name} to {data_point_dir}")
                with zipfile.ZipFile(subzip_path, 'r') as subzip:
                    subzip.extractall(data_point_dir)
            else:
                if verbose:
                    print(f"Sub-zip {subzip_path.name} already unzipped")

            data_point_dirs.append(data_point_dir)

        return data_point_dirs

    def split_data(self, num_splits: int, train_ratio: float = 0.8, stratified: bool = True):
        """Splits the data into stratified folds."""
        # get all the labels
        labels = []
        for data_dir in self.data_dirs:
            json_label_path = data_dir/'label.json'
            with open(json_label_path, 'r') as f:
                label_data = json.load(f)
            label = self.LABEL_MAP[label_data['label']]
            labels.append(label)

        # split data using train_test_split
        train_splits, val_splits, test_splits = [], [], []
        for i in range(num_splits):
            stratify = labels if stratified else None
            train_data, test_data, train_labels, test_labels = train_test_split(
                self.data_dirs,
                labels,
                train_size=train_ratio,
                shuffle=True,
                stratify=stratify,
                random_state=i
            )

            stratify = test_labels if stratified else None
            val_data, test_data = train_test_split(
                test_data,
                train_size=0.5,
                shuffle=True,
                stratify=stratify,
                random_state=i
            )
            train_splits.append(train_data)
            val_splits.append(val_data)
            test_splits.append(test_data)

        if num_splits == 1:
            return train_splits[0], val_splits[0], test_splits[0]
        else:
            return train_splits, val_splits, test_splits

    def load_data(
        self,
        train_paths: list[Path],
        val_paths: list[Path],
        test_paths: list[Path]
    ) -> tuple[list[ClientData], list[ClientData], list[ClientData]]:
        """Returns the data as a JSON object."""
        json_data_filenames = [
            'account_form.json',
            'client_description.json',
            'client_profile.json',
            'passport.json'
        ]

        def get_json_and_labels(paths: list[Path]):
            all_client_data = []
            for path in paths:
                json_data = {}

                # add the path to the json_data
                json_data['client_file'] = str(path)

                # load the client data, which __has__ to exist
                for filename in json_data_filenames:
                    filename_no_ext = filename.split('.')[0]  # remove extension (".json")
                    with open(path/filename, 'r') as f:
                        json_data[filename_no_ext] = json.load(f)
                    if filename_no_ext in ['account_form', 'client_profile']:
                        if isinstance(json_data[filename_no_ext]['passport_number'], list):
                            json_data[filename_no_ext]['passport_number'] = json_data[filename_no_ext]['passport_number'][0]

                # loads the label from the label.json file, __if__ it exists
                if (path/'label.json').exists():
                    with open(path/'label.json', 'r') as f:
                        label_data = json.load(f)
                        json_data['label'] = self.LABEL_MAP[label_data['label']]

                # create a ClientData object from the dict
                client_data = ClientData(**json_data)
                all_client_data.append(client_data)

            return all_client_data

        train_data = get_json_and_labels(train_paths)
        val_data = get_json_and_labels(val_paths)
        test_data = get_json_and_labels(test_paths)

        return train_data, val_data, test_data

    def save_split_data_pickle(
        self,
        split_dir: Path,
        train_data: list[ClientData],
        val_data: list[ClientData],
        test_data: list[ClientData]
    ):
        """Saves the split data to a pickle file."""
        split_dir.mkdir(parents=True, exist_ok=True)
        with open(split_dir/'train.pkl', 'wb') as f:
            pickle.dump(train_data, f)
        with open(split_dir/'val.pkl', 'wb') as f:
            pickle.dump(val_data, f)
        with open(split_dir/'test.pkl', 'wb') as f:
            pickle.dump(test_data, f)


if __name__ == '__main__':
    data_pipeline = DataPipeline(ROOT_DIRS)  # unzips (if not already unzipped) and gets data paths
    # print(f"Data directories: {data_pipeline.data_dirs}")  # prints the data directories

    # CV
    train_paths, val_paths, test_paths = data_pipeline.split_data(num_splits=1, train_ratio=0.8, stratified=True)
    print(f'Loaded {len(train_paths)} training paths, {len(val_paths)} validation paths, and {len(test_paths)} test paths.')
    # get dataclasses
    train_data, val_data, test_data = data_pipeline.load_data(
        train_paths, val_paths, test_paths
    )
    # save the split data to pickle files
    data_pipeline.save_split_data_pickle(
        SPLIT_DATA_SPLIT1_DIR,
        train_data,
        val_data,
        test_data
    )

    # 5-fold CV
    train_paths, val_paths, test_paths = data_pipeline.split_data(num_splits=5, train_ratio=0.8, stratified=True)
    for i, (trp, v, tep) in enumerate(zip(train_paths, val_paths, test_paths)):
        print(f'Fold {i+1}:')
        print(f'Loaded {len(trp)} training paths, {len(v)} validation paths, and {len(tep)} test paths.')
        # get jsons and labels
        train_data, val_data, test_data = data_pipeline.load_data(
            trp, v, tep
        )
        # save the split data to pickle files
        data_pipeline.save_split_data_pickle(
            SPLIT_DATA_SPLIT5_DIR/f'split_{i+1}',
            train_data,
            val_data,
            test_data
        )
