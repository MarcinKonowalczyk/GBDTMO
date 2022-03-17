import numpy as np

import os, sys

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), ".")))

from gbdtmo.sklearn import GBDTMO

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
common_params = dict(shape=(history, future), verbose=True, max_iter=100)
# search_params = dict(correlated=(True, False), learning_rate=(0.1,0.2))
# search_params = dict(learning_rate=(0.1, 0.2, 0.3, 0.4), max_depth=(1,2,3,4))
search_params = dict(seed=(0, 1))

booster = GBDTMO(**common_params)
# booster.fit(X_train, y_train)

clf = GridSearchCV(booster, search_params)
# clf.fit(X_train, y_train)

# yp = booster.predict(X_test)

# print(f"score = {booster.score(X_test, y_test)}")
