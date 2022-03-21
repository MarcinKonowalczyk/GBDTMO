import numpy as np

import os, sys

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), ".")))

from gbdtmo import Loss
from gbdtmo.sklearn import GBDTRegressor

# Generate data
from generate_timeseries import generate

data = generate(30 * 48)

from sklearn.model_selection import train_test_split
# Prepare timeseries data for the booster
history = 48  # Data points to use as explanatory variables
future = 12
from more_itertools import windowed

temp = np.array(list(windowed(data, 2 * history)))
X = temp[:, :history]  # past
y = temp[:, history:(history + future)]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# HPO
from sklearn.model_selection import GridSearchCV

# TODO: Early stop and set_eval
common_params = dict(
    shape=(history, future), verbose=False, max_iter=2000, early_stop=10, eval_fraction=0.2)
search_params = dict(
    correlated=(False,True),
    learning_rate=tuple(np.linspace(0, 1, 11)[1:]),
    max_depth=tuple(np.arange(1, 7)),
    reg_l1=tuple(np.linspace(0, 1, 11)),
    reg_l2=tuple(np.linspace(0, 1, 11)[1:]),
    topk=(1, 2, 4, 8, future),
)
# search_params = dict(
#     correlated=(False,),
#     learning_rate=(0.8,),
#     max_depth=(5,),
#     reg_l1=(0.8,),
#     reg_l2=(0.3,),
#     topk=(2,),
# )
# search_params = dict(seed=(0, 1))

from skopt import BayesSearchCV

booster = GBDTRegressor(**common_params)
# clf = GridSearchCV(booster, search_params, verbose=3)
clf = BayesSearchCV(booster, search_params, verbose=3, n_iter=1024)
# booster.fit(X_train, y_train)
clf.fit(X_train, y_train)
# yp = booster.predict(X_test)

# print(f"score = {booster.score(X_test, y_test)}")
