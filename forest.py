from load_dataset import full_load, loaders
from sklearn import svm
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = None
RF_ITERATIONS = 100
SVM_ITERATIONS = 24
TEST_SIZE = 0.2
PARAMETERS = {
    "criterion": ["gini", "entropy"],
    "n_estimators": [i for i in range(100, 1000)],
    "max_depth": [None] + [i for i in range(2, 20)],
    "min_samples_split": [i for i in range(2, 10)],
}
SVM_PARAMETERS = {
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    "degree": [i for i in range(6)],
}


class Classifier:
    def __init__(self, type_="forest", params={}, **kwargs):
        """
        Creates classifier of specified type.
        """
        full_params = {
            "forest": {
                "random_state": RANDOM_STATE,
                "n_jobs": -1,
                "criterion": "gini",
                "n_estimators": 100,
            },
            "svm": {"kernel": "linear", "decision_function_shape": "ovr"},
        }[type_]
        full_params.update(**params, **kwargs)

        self._classifier = {"forest": RandomForestClassifier, "svm": svm.SVC}[type_](
            **full_params
        )

        self._best_params = full_params

    def search(self, iterations, parameters, features, labels, apply_best=True):
        """
        Create RandomForest, then find best parameters with RandomizedSearch,
        what is great, we don't have to split our data, because RandomizedSearch
        has implemented cross-validation feature. Basically, default value (5-fold)
        makes sure, that we run 5 iterations of fitting algorithm, each one with other
        train and test values, and score function is run on test data. So, test size
        each time is equal to 20% of all data size. Also, cross_validate uses random state,
        by default 100, wchich is awfully enough for us.
        """
        self._gs = RandomizedSearchCV(
            self._classifier,
            parameters,
            n_jobs=-1,
            n_iter=iterations,
            random_state=RANDOM_STATE,
        )
        self._search = self._gs.fit(features, labels)
        self._best_params = self._search.best_params_
        if apply_best:
            self._classifier = self._search

    def __getattr__(self, attrname):
        """
        Just to return functions such as fit, score etc
        """
        return getattr(self._classifier, attrname)

    @property
    def best_params(self):
        return self._best_params

    @property
    def best_score(self):
        return self._search.best_score_

    @staticmethod
    def split(features, labels):
        return train_test_split(
            features, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

    def report(self, param_name, num_results=15):
        from matplotlib import pyplot as plt
        from IPython.display import display
        import pandas as pd

        clf = self._search.best_estimator_
        clf_params = self._search.best_params_
        clf_score = self._search.best_score_
        clf_stdev = self._search.cv_results_["std_test_score"][self._search.best_index_]
        cv_results = self._search.cv_results_

        print("best parameters: {}".format(clf_params))
        print("best score:      {:0.5f} (+/-{:0.5f})".format(clf_score, clf_stdev))

        import pprint

        pprint.pprint(clf.get_params())

        # pick out the best results
        scores_df = pd.DataFrame(cv_results).sort_values(by="rank_test_score")

        best_row = scores_df.iloc[0, :]
        best_mean = best_row["mean_test_score"]
        best_stdev = best_row["std_test_score"]
        best_param = best_row["param_" + param_name]

        # display the top 'num_results' results
        display(
            pd.DataFrame(cv_results).sort_values(by="rank_test_score").head(num_results)
        )

        # plot the results
        scores_df = scores_df.sort_values(by="param_" + param_name)

        means = scores_df["mean_test_score"]
        stds = scores_df["std_test_score"]
        params = scores_df["param_" + param_name]

        plt.figure(figsize=(8, 8))
        plt.errorbar(params, means, yerr=stds)

        plt.axhline(y=best_mean + best_stdev, color="red")
        plt.axhline(y=best_mean - best_stdev, color="red")
        plt.plot(best_param, best_mean, "or")

        plt.title(param_name + " vs Score\nBest Score {:0.5f}".format(clf_score))
        plt.xlabel(param_name)
        plt.ylabel("Score")
        plt.show()


"""
Avalibe parameters for training are from parameters section there:
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
"""
