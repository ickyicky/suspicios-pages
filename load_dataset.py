import os
import pandas as pd
import numpy as np

from scipy.io import arff
from io import StringIO


def load_dataset(path):
    """
    Loads arff dataset from given path
    """
    buff = None

    with open(path, "r") as f:
        buff = StringIO(f.read())

    assert buff is not None
    data, meta = arff.loadarff(buff)
    return data, meta


def find_dataset(path_="."):
    """
    Finds all arff dataset in specified folder
    """
    return [x for x in os.listdir(path_) if x.lower().endswith("arff")]


def normalize(data):
    """
    Normalizes data to int dataformat
    """
    return pd.DataFrame(data).astype("int32")


def prepare(features):
    """
    Prepares features and labels.

    Extract features, feature_list and labels from given data
    """
    labels = np.array(features["Result"])
    features = features.drop("Result", axis=1)
    feature_list = list(features.columns)
    features = np.array(features)
    return features, feature_list, labels


def full_load():
    """
    Finds local dataset, then loads and prepares it
    """
    data, meta = load_dataset(find_dataset()[0])
    data = normalize(data)
    return prepare(data)


def load_yaml(path):
    import yaml

    return yaml.load(path)


def load_py(path):
    content = None
    with open(path, "r") as f:
        content = f.read()

    return eval(content)


def load_json(path):
    import json

    content = None
    with open(path, "r") as f:
        content = f.read()

    return json.loads(content)


loaders = {
    ".py": load_py,
    ".yml": load_yaml,
    ".yaml": load_yaml,
    ".json": load_json,
}
