import pandas as pd
import yaml


def load_config(path="config/config.yaml"):

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config


def load_data(config):

    file_path = config["data"]["path"]

    df = pd.read_csv(file_path)

    return df


def validate_data(df):

    if df.empty:
        raise ValueError("Dataset is empty")

    if df.isnull().all().any():
        raise ValueError("Some columns are entirely null")

    return True