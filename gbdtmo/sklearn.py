import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from . import *

# =============================================================================================
#
#   ####  ##  ##  ##      #####    ###    #####    ##     ##
#  ##     ## ##   ##      ##      ## ##   ##  ##   ####   ##
#   ###   ####    ##      #####  ##   ##  #####    ##  ## ##
#     ##  ## ##   ##      ##     #######  ##  ##   ##    ###
#  ####   ##  ##  ######  #####  ##   ##  ##   ##  ##     ##
#
# =============================================================================================

# sklearn-friendly wrapper for the booster


class GBDTRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        *,
        max_depth=5,
        max_leaves=32,
        seed=0,
        min_samples=5,
        learning_rate=0.2,
        reg_l1=0.0,
        reg_l2=1.0,
        gamma=1e-3,
        early_stop=0,
        verbose=True,
        max_caches=16,
        topk=0,
        one_side=True,
        max_bins=32,
        alpha=0.0,
        eval_fraction=0.0,
        correlated=True,
        max_iter=100,
    ) -> None:
        super().__init__()
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.seed = seed
        self.min_samples = min_samples
        self.learning_rate = learning_rate
        self.reg_l1 = reg_l1
        self.reg_l2 = reg_l2
        self.gamma = gamma
        self.early_stop = early_stop
        self.verbose = verbose
        self.max_caches = max_caches
        self.topk = topk
        self.one_side = one_side
        self.max_bins = max_bins
        self.alpha = alpha
        self.eval_fraction = eval_fraction
        self.correlated = correlated
        self.max_iter = max_iter

    def __getstate__(self):
        state = super().__getstate__()
        if hasattr(self, "booster_"):
            _state = self.booster_.get_state()
            state["tree_array"], state["threshold_array"], state["leaf_array"] = _state
        return state

    def __setattr__(self, name, value):
        if hasattr(self, "booster_") and (name in self.get_params() or name == "booster_"):
            raise TypeError(
                f"Cannot change attribute '{name}' of a fitted estimator. Use `base.clone()`, change the attributes and refit."
            )
        super().__setattr__(name, value)

    def fit(self, X, y):
        X, y = check_X_y(X, y, multi_output=True)

        y = y.reshape(-1, 1) if len(y.shape) == 1 else y
        self.shape = (X.shape[1], y.shape[1])

        self.booster_ = (GBDTMulti if self.correlated else GBDTSingle)(self.shape, self.__dict__ | dict(loss=Loss.mse))
        self.booster_.set_data_regression(X, y)
        self.booster_.train(self.max_iter)

        self.n_trees_ = self.booster_._lib_GetNTrees()  # hacky

        return self

    def predict(self, X):
        check_is_fitted(self, "booster_")
        X = check_array(X)
        return self.booster_.predict(X)


# =======================================================================================================================================
#
#  #####    #####   ####        #####   #####   #####     ####    ###     ####  ######  #####  #####
#  ##  ##   ##     ##           ##     ##   ##  ##  ##   ##      ## ##   ##       ##    ##     ##  ##
#  #####    #####  ##           #####  ##   ##  #####    ##     ##   ##   ###     ##    #####  #####
#  ##  ##   ##     ##           ##     ##   ##  ##  ##   ##     #######     ##    ##    ##     ##  ##
#  ##   ##  #####   ####        ##      #####   ##   ##   ####  ##   ##  ####     ##    #####  ##   ##
#
# =======================================================================================================================================


class GBDTRecursiveForcaster(GBDTRegressor):
    def __init__(
        self,
        *,
        max_depth=5,
        max_leaves=32,
        seed=0,
        min_samples=5,
        learning_rate=0.2,
        reg_l1=0.0,
        reg_l2=1.0,
        gamma=1e-3,
        early_stop=0,
        verbose=True,
        max_caches=16,
        topk=0,
        one_side=True,
        max_bins=32,
        alpha=0.0,
        eval_fraction=0.0,
        correlated=True,
        max_iter=100,
        future_increment=None,
    ) -> None:
        super().__init__()
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.seed = seed
        self.min_samples = min_samples
        self.learning_rate = learning_rate
        self.reg_l1 = reg_l1
        self.reg_l2 = reg_l2
        self.gamma = gamma
        self.early_stop = early_stop
        self.verbose = verbose
        self.max_caches = max_caches
        self.topk = topk
        self.one_side = one_side
        self.max_bins = max_bins
        self.alpha = alpha
        self.eval_fraction = eval_fraction
        self.correlated = correlated
        self.max_iter = max_iter
        self.future_increment = future_increment

    def fit(self, X, y):
        X, y = check_X_y(X, y, multi_output=True)

        y = y.reshape(-1, 1) if len(y.shape) == 1 else y
        self.future_length = y.shape[1]
        if self.future_increment is None:
            self.shape = (X.shape[1], y.shape[1])
        elif self.future_increment <= y.shape[1]:
            self.shape = (X.shape[1], self.future_increment)
            y = y[:, : self.future_increment]
        elif self.future_increment > y.shape[1]:
            raise TypeError("Future increment larger than the size of the training data.")

        booster_object = GBDTMulti if self.correlated else GBDTSingle
        params = self.__dict__ | dict(loss=Loss.mse)

        self.booster_ = booster_object(self.shape, params)
        self.booster_.set_data_regression(X, y)
        self.booster_.train(self.max_iter)

        self.n_trees_ = self.booster_._lib_GetNTrees()  # hacky

        return self

    def predict(self, X, future_length=None):
        check_is_fitted(self, "booster_")
        X = check_array(X)
        assert X.shape[1] == self.shape[0]
        inp_dim = self.shape[0]
        # By default use the same future length as for training
        if not future_length:
            future_length = self.future_length

        yp = self.booster_.predict(X)
        X_incr = X.copy()
        while yp.shape[1] < future_length:
            i = max(inp_dim - yp.shape[1], 0)
            X_slice, yp_slice = (
                range(inp_dim - i, inp_dim),
                range(yp.shape[1] - (inp_dim - i), yp.shape[1]),
            )
            X_incr = np.c_[X[:, X_slice], yp[:, yp_slice]]
            assert X_incr.shape[1] == inp_dim
            yp_incr = self.booster_.predict(X_incr)
            yp = np.c_[yp, yp_incr]
        yp = yp[:, :future_length]

        return yp


# ===============================================================================================================================================================
#
#  #####  #####  ######        ######  #####      ###    ##     ##   ####  #####   #####   #####    ###    ###  #####  #####
#  ##     ##       ##            ##    ##  ##    ## ##   ####   ##  ##     ##     ##   ##  ##  ##   ## #  # ##  ##     ##  ##
#  #####  #####    ##            ##    #####    ##   ##  ##  ## ##   ###   #####  ##   ##  #####    ##  ##  ##  #####  #####
#  ##     ##       ##            ##    ##  ##   #######  ##    ###     ##  ##     ##   ##  ##  ##   ##      ##  ##     ##  ##
#  ##     ##       ##            ##    ##   ##  ##   ##  ##     ##  ####   ##      #####   ##   ##  ##      ##  #####  ##   ##
#
# ===============================================================================================================================================================

from sklearn.base import BaseEstimator, TransformerMixin


class FFTTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, *, trim=0.0) -> None:
        super().__init__()
        self.trim = trim

    def fit(self, X, y=None):
        self._validate_data(X, reset=True, accept_sparse=False, ensure_2d=True)
        # self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = self._validate_data(X, reset=False, accept_sparse=False, ensure_2d=True)
        n_samples, n_features = X.shape

        n_fft_features = int((n_features // 2 + 1) * (1 - self.trim))
        XFFT = np.zeros((n_samples, 2 * n_fft_features), dtype=X.dtype)
        for i in range(n_samples):
            XX = np.fft.rfft(X[i, :])
            XX = XX[:n_fft_features]
            XFFT[i, :] = np.r_[np.real(XX), np.imag(XX)]

        return XFFT

    def inverse_transform(self, XFFT):
        n_samples, n_fft_features = XFFT.shape
        n_fft_features = n_fft_features // 2
        X = np.zeros((n_samples, self.n_features_in_), dtype=XFFT.dtype)
        for i in range(n_samples):
            XX = XFFT[i, :n_fft_features] + 1j * XFFT[i, n_fft_features:]
            X[i, :] = np.fft.irfft(XX, self.n_features_in_)

        return X
