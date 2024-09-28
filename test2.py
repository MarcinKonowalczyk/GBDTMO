import os
import sys

import numpy as np

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), ".")))

# from gbdtmo import Loss
from gbdtmo.sklearn import GBDTRecursiveForcaster

# Generate data
from generate_timeseries import generate

data = generate(30 * 48)

from sklearn.model_selection import train_test_split

# Prepare timeseries data for the booster
history = 48  # Data points to use as explanatory variables
future = 2 * 48
from more_itertools import windowed

temp = np.array(list(windowed(data, history + future)))
X = temp[:, :history]  # past
y = temp[:, history : (history + future)]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# ============================================================
#
#  ##   ##  #####    #####
#  ##   ##  ##  ##  ##   ##
#  #######  #####   ##   ##
#  ##   ##  ##      ##   ##
#  ##   ##  ##       #####
#
# ============================================================

# from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Integer, Real

common_params = dict(
    verbose=False,
    max_iter=1000,
    early_stop=10,
    eval_fraction=0.2,
    correlated=True,
    reg_l2=1.0,
)

search_params = dict(
    future_increment=(1, 2, 4, 6, 8, 12, 24, 48),
    learning_rate=Real(0.2, 1.0),
    max_depth=Integer(1, 4),
    gamma=Real(1e-4, 1e-1, prior="log-uniform"),
    reg_l1=Real(0.0, 10.0),
    topk=Integer(1, 48),
)

# OrderedDict([('future_increment', 47),
#              ('learning_rate', 0.1),
#              ('max_depth', 7),
#              ('reg_l1', 1.0),
#              ('reg_l2', 1.0),
#              ('topk', 39)])

from functools import partial

from sklearn.metrics import make_scorer, mean_squared_error, r2_score


def make_slice_score(f, *, slice):
    def slice_score(y_true, y_pred, **kwargs):
        y_true, y_pred = y_true[:, slice], y_pred[:, slice]
        return f(y_true, y_pred, **kwargs)

    return slice_score


_1da = partial(make_slice_score, slice=range(0, 48))
_2da = partial(make_slice_score, slice=range(48, 96))

scoring = {
    "score": make_scorer(_1da(r2_score)),
    "r2_1da": make_scorer(_1da(r2_score)),
    "r2_2da": make_scorer(_2da(r2_score)),
    "mse_1da": make_scorer(_1da(mean_squared_error)),
    "mse_2da": make_scorer(_2da(mean_squared_error)),
}

booster = GBDTRecursiveForcaster(**common_params)
booster.fit(X_train, y_train)
yp = booster.predict(X_test)

# hpo = GridSearchCV(booster, search_params, scoring=scoring, refit="R2", verbose=3)
hpo = BayesSearchCV(booster, search_params, scoring=scoring, refit="score", verbose=2, n_jobs=5)
hpo.fit(X_train, y_train)
# yp = booster.predict(X_test)

# print(f"score = {booster.score(X_test, y_test)}")
