# suspicios-pages
Random forest model fit to predict if page is suspicious

to try this out import some dataset, for example from:
https://archive.ics.uci.edu/ml/datasets/Website+Phishing

copy the file into this folder and simply run with:
python forest.py

Custom kwargs for training rf are just simple scikit parameters,
listed under parameters section there:
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html