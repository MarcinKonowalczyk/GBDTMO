import numpy as np

import os, sys

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), ".")))

# from gbdtmo import Loss
from gbdtmo.sklearn import GBDTRegressor, FFTTransformer

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
y = temp[:, history:(history + future)]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

#============================================================
#
#  ##   ##  #####    #####
#  ##   ##  ##  ##  ##   ##
#  #######  #####   ##   ##
#  ##   ##  ##      ##   ##
#  ##   ##  ##       #####
#
#============================================================

# from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical

common_params = dict(
    verbose=False,
    max_iter=1000,
    early_stop=10,
    eval_fraction=0.2,
    correlated=True,
    reg_l2=1.0,
)

search_params = dict(
    gbdt__learning_rate=Real(0.2, 1.0),
    gbdt__max_depth=Integer(1, 4),
    gbdt__gamma=Real(1e-4, 1e-1, prior="log-uniform"),
    gbdt__reg_l1=Real(0.0, 10.0),
    gbdt__topk=Integer(1, 48),
)

# OrderedDict([('future_increment', 47),
#              ('learning_rate', 0.1),
#              ('max_depth', 7),
#              ('reg_l1', 1.0),
#              ('reg_l2', 1.0),
#              ('topk', 39)])

from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from functools import partial


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

# from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    # StandardScaler(),
    ('fft', FFTTransformer(trim=0.8)),
    ('gbdt', GBDTRegressor(**common_params)),
])

pipe.fit(X_train, y_train)
yp = pipe.predict(X_test)

exit(1)
# hpo = GridSearchCV(booster, search_params, scoring=scoring, refit="R2", verbose=3)
hpo = BayesSearchCV(pipe, search_params, scoring=scoring, refit="score", verbose=3, n_jobs=5)
hpo.fit(X_train, y_train)
yp = hpo.predict(X_test)

# Plot
from skopt.plots import plot_objective, plot_histogram

plot_objective(hpo.optimizer_results_[0])
