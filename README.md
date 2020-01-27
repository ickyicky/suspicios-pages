# suspicios-pages
Random forest model fit to predict if page is suspicious

example dataset is downloaded from:
https://archive.ics.uci.edu/ml/datasets/Website+Phishing

copy the file into this folder and simply run with:
python3 main.py

help is accesable via:
python3 main.py --help

Avalibe options:
- params FILE - specify file with stored params, you wan't to use for RF classifier. It is advised not to adjust parameters after loading yours afterwards
- predict PREDICT - after training predict result for those params
- iterations ITERATIONS - specify number of iterations for RF automatic parameter adjustment algorithm
- svm-iterations SVM_ITERATIONS - as above, but for SVM
- dont-search - don't adjust parameters, leave defaults or specified, for RF
- dont-search-svm - as above, but for SVM
- console - run simple shell afterwards, this way you can access forest (RF classifier) and svc (SVM classifier) instances
- report PARAM - display report for given parameter (only works if was automatically adjusted), for RF
- report-svm PARAM - as above, but for svm

Custom kwargs for training rf are just simple scikit parameters,
listed under parameters section there:
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html