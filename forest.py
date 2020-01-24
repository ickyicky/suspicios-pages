from load_dataset import full_load, loaders
from sklearn import svm
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = None
RANDOM_ITERATIONS = 100
TEST_SIZE = 0.25
PARAMETERS = {
    "criterion": ["gini", "entropy"],
    "n_estimators": [i for i in range(100, 1000)],
    "max_depth": [None] + [i for i in range(2, 20)],
    "min_samples_split": [i for i in range(2, 10)],
}


def create_svm_reference(train_features, test_features, train_labels, test_labels):
    """
    Basic SVM
    """
    clf = svm.SVC(kernel="linear")
    clf.fit(train_features, train_labels)
    return clf, clf.score(test_features, test_labels)


def train(features, labels, parameters=PARAMETERS, iterations=RANDOM_ITERATIONS):
    """
    Create RandomForest, then find best parameters with RandomizedSearch,
    what is great, we don't have to split our data, because RandomizedSearch
    has implemented cross-validation feature. Basically, default value (5-fold)
    makes sure, that we run 5 iterations of fitting algorithm, each one with other
    train and test values, and score function is run on test data. So, test size
    each time is equal to 20% of all data size. Also, cross_validate uses random state,
    by default 100, wchich is awfully enough for us.
    """
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    gs = RandomizedSearchCV(
        rf, parameters, n_jobs=-1, n_iter=iterations, random_state=RANDOM_STATE
    )
    search = gs.fit(features, labels)

    # manual test, just to validate our results
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    # score, basically calls predict and then validates result
    accuracy = search.score(test_features, test_labels)

    print("Best params:", search.best_params_)
    print("Accuracy:", accuracy)

    ref, ref_score = create_svm_reference(
        train_features, test_features, train_labels, test_labels
    )
    print("Reference score:", ref_score)
    print("Difference:", accuracy - ref_score)

    return search, accuracy


"""
Avalibe parameters for training are from parameters section there:
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
"""

if __name__ == "__main__":
    features, feature_list, labels = full_load()

    import re
    import sys

    try:
        params_path = sys.argv[1]
    except IndexError:
        params_path = input("Specify path to file with stored parameters values: ")

    if params_path:
        try:
            loader = loaders[re.findall(r"(\.[\w\d]+)$", params_path)[0]]
        except (KeyError, IndexError):
            raise Exception(
                f"Unsuported file format ! Supported file formats: {''.join(loaders.keys())}"
            )

        parameters = loader(params_path)
    else:
        print("Running with default params.")
        parameters = PARAMETERS

    try:
        iterations = int(sys.argv[2])
    except (IndexError, ValueError):
        try:
            iterations = int(input("Specify iterations number: "))
        except ValueError:
            print("Running with default iteration number.")
            iterations = RANDOM_ITERATIONS

    rf, accuracy = train(features, labels, parameters, iterations)
