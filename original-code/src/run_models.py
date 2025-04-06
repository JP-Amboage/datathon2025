from consts import ROOT_ZIPFILES
from src.base_data import DataPipeline
from src.predictor import RandomPredictor


def main():


    pipeline = DataPipeline(ROOT_ZIPFILES)
    # CV
    train_paths, val_paths, test_paths = pipeline.split_data(num_splits=1, train_ratio=0.8, stratified=True)
    print(
        f'Loaded {len(train_paths)} training paths, {len(val_paths)} validation paths, and {len(test_paths)} test paths.')

    train_data, val_data, test_data = pipeline.load_data(
        train_paths, val_paths, test_paths
    )

    model = RandomPredictor()

    preds = model.predict(val_data)

    print(model.get_scores(preds, [data.label for data in val_data]))



if __name__ == '__main__':
    main()