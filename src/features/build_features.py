import pathlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from features_definitions import feature_build

def load_data(data_path):
    # Load your dataset from a given path
    df = pd.read_csv(data_path)
    return df

def save_data(data, output_path):
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path + '/Cab_Data.csv', index=False)


if __name__ == "__main__":
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    data_path = home_dir.as_posix() + '/data/raw/Cab_Data.csv'

    data = pd.read_csv(data_path)

    output_path = home_dir.as_posix() + '/data/processed'

    data = feature_build(data, 'Cab_Data')

    save_data(data, output_path)