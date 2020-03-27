from sklearn import ensemble
from sklearn import linear_model

MODELS = {
    "randomforest": ensemble.RandomForestClassifier(n_estimators = 150, n_jobs = -1, verbose = 2),
    "extratrees": ensemble.ExtraTreesClassifier(n_estimators = 100, n_jobs = -1, verbose = 2),
    "logistic_regression": linear_model.LogisticRegression()
}