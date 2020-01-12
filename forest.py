import numpy as np
from load_dataset import full_load
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

RANDOM_STATE = 25
TEST_SIZE = 0.15
DEFAULT_KWARGS = {
    "n_estimators": 500,
    "criterion": "mae",
}


def prepare_all():
    features, feature_list, labels = full_load()

    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print("Training Features Shape:", train_features.shape)
    print("Training Labels Shape:", train_labels.shape)
    print("Testing Features Shape:", test_features.shape)
    print("Testing Labels Shape:", test_labels.shape)
    return train_features, test_features, train_labels, test_labels


def train(train_features, test_features, train_labels, test_labels, **kwargs):
    """
    Create RandomForest, fit it and then check accuracy on given test features.

    Additional kwargs are passed as RandomForestRegressor arameters
    """
    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **kwargs)
    rf.fit(train_features, train_labels)

    predictions = rf.predict(test_features)
    correct = np.sum(predictions == test_labels)
    accuracy = 1 - (correct / len(test_labels))

    print("Kwargs:", kwargs)
    print("Currect:", correct)
    print("Amount:", len(test_labels))
    print("Accuracy:", accuracy)
    return rf, accuracy


"""
At this moment we can try to fit our forest. With 500 estimators and mean absolute error criterion
accuracy reaches 70%, co it's not bad. example:

train(
    train_features,
    test_features,
    train_labels,
    test_labels,
    n_estimators=500,
    criterion="mae",
)

Avalibe kwargs for training are under parameters section there:
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
"""


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

if __name__ == "__main__":
    train_features, test_features, train_labels, test_labels = prepare_all()

    import re

    kwargs_path = input("Specify path to file with stored kwargs: ")

    if kwargs_path:
        try:
            loader = loaders[re.findall(r"(\.[\w\d]+)$", kwargs_path)[0]]
        except (KeyError, IndexError):
            raise Exception(
                f"Unsuported file format ! Supported file formats: {''.join(loaders.keys())}"
            )

        kwargs = loader(kwargs_path)
        if not isinstance(kwargs, (list, tuple)):
            kwargs = (kwargs,)
    else:
        print("Running with default kwargs.")
        kwargs = [DEFAULT_KWARGS]

    best_rf, best_accuracy = None, 0.0

    for k in kwargs:
        rf, accuracy = train(
            train_features, test_features, train_labels, test_labels, **k
        )

        if accuracy > best_accuracy:
            print("Found new best accuracy!")
            best_rf, best_accuracy = rf, accuracy

    print("Best accuracy:", best_accuracy)
