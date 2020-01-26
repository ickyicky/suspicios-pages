from forest import full_load, RF_ITERATIONS, SVM_ITERATIONS, PARAMETERS, SVM_PARAMETERS, loaders, Classifier
import argparse
import re

features, feature_list, labels = full_load()

parser = argparse.ArgumentParser(
    description="Random Forest (RF) searching for suspicious pages",
    prog="python main.py",
)
parser.add_argument(
    "--params", help="Specify file with RF parameters", metavar="FILE"
)
parser.add_argument(
    "--iterations",
    help="Specify number of iterations for automatical adjustment",
    action="store",
    default=RF_ITERATIONS,
    type=int,
)
parser.add_argument(
    "--svm-iterations",
    help="Specify number of iterations for SVM automatical adjustment",
    action="store",
    default=SVM_ITERATIONS,
    type=int,
)
parser.add_argument(
    "--svm-dont-search",
    help="Specify, if svm shouldn't be automatically adjusted",
    action="store_true",
)
parser.add_argument(
    "--dont-search",
    help="RF shouldn't be adjusted automatically",
    action="store_true",
)
parser.add_argument(
    "--console", help="Run very primitive console afterwards", action="store_true"
)
parser.add_argument(
    "--report",
    nargs="+",
    help="Run report of given parameters for RF afterwards.}",
    choices=PARAMETERS.keys(),
    action="store",
)
parser.add_argument(
    "--report-svm",
    nargs="+",
    help="Run report of given parameters for SVM afterwards",
    choices=SVM_PARAMETERS.keys(),
    action="store",
)
args = parser.parse_args()

if args.params:
    try:
        loader = loaders[re.findall(r"(\.[\w\d]+)$", args.params)[0]]
    except (KeyError, IndexError):
        raise Exception(
            f"Unsuported file format ! Supported file formats: {', '.join(loaders.keys())}"
        )

    parameters = loader(args.params)
else:
    parameters = {}
    print("Running with default params:", parameters)

train_features, test_features, train_labels, test_labels = Classifier.split(
    features, labels
)

forest = Classifier(params=parameters)
forest.fit(train_features, train_labels)
forest_score = forest.score(test_features, test_labels)
print("RF accuracy:", forest_score)

if not args.dont_search:
    forest.search(args.iterations, PARAMETERS, features, labels)
    print("RF best params:", forest.best_params)
    forest_score = forest.score(test_features, test_labels)
    print("RF accuracy after search:", forest_score)

svc = Classifier("svm")
svc.fit(train_features, train_labels)
svm_score = svc.score(test_features, test_labels)
print("SVM accuracy:", svm_score)

if not args.svm_dont_search:
    svc.search(args.svm_iterations, SVM_PARAMETERS, features, labels)
    print("SVM best params:", svc.best_params)
    svm_score = svc.score(test_features, test_labels)
    print("SVM accuracy after adjusting:", svm_score)

print("Gain:", forest_score - svm_score)

if args.report:
    for p in args.report:
        forest.report(p)

if args.report_svm:
    for p in args.report_svm:
        svc.report(p)

if args.console:
    print("To exit, simply type exit")

    while True:
        in_ = input("")

        if in_ == "exit":
            break

        try:
            result = exec(in_)
        except Exception as e:
            print(str(e))
