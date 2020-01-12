import numpy as np
from load_dataset import full_load
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

RANDOM_STATE = 25
TEST_SIZE = 0.15

features, feature_list, labels = full_load()

train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

print("Training Features Shape:", train_features.shape)
print("Training Labels Shape:", train_labels.shape)
print("Testing Features Shape:", test_features.shape)
print("Testing Labels Shape:", test_labels.shape)


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


train(
    train_features,
    test_features,
    train_labels,
    test_labels,
    n_estimators=500,
    criterion="mae",
)

"""
At this moment we can try to fit our forest better, with 500 estimators and mean absolute error criterion
accuracy reaches 70%, co it's not bad.
"""
