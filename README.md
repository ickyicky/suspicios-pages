# suspicios-pages
Random forest model fit to predict if page is suspicious

example dataset is downloaded from:
https://archive.ics.uci.edu/ml/datasets/Website+Phishing

copy the file into this folder and simply run with:
python3 main.py

help is accesable via:
python3 main.py --help

Custom kwargs for training rf are just simple scikit parameters,
listed under parameters section there:
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html