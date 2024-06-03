import pandas as pd


def load_dataframe(dataset_path):
    dataset = pd.read_csv(dataset_path)

    return dataset
