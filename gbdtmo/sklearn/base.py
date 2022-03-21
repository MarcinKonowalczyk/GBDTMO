from ..gbdtmo import GBDTMulti, GBDTSingle, Loss, HyperParameters

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

HP_NAMES = list(zip(*HyperParameters._fields_))[0]

class GBDTRegressor(BaseEstimator, RegressorMixin):

    shape = None

    def __init__(
        self,
        shape=(1, 1),
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
        correlated=True,
        max_iter=100,
        alpha=0.0,
        eval_fraction=0.0,
    ) -> None:
        super().__init__()
        self.shape = shape
        self.correlated = correlated
        self.max_iter = max_iter
        self.loss = Loss.mse;
        self.max_depth = max_depth;
        self.max_leaves = max_leaves;
        self.seed = seed;
        self.min_samples = min_samples;
        self.learning_rate = learning_rate;
        self.reg_l1 = reg_l1;
        self.reg_l2 = reg_l2;
        self.gamma = gamma;
        self.early_stop = early_stop;
        self.verbose = verbose;
        self.max_caches = max_caches;
        self.topk = topk;
        self.one_side = one_side;
        self.max_bins = max_bins;
        self.alpha = alpha;
        self.eval_fraction = eval_fraction;

    def __getstate__(self):
        state = super().__getstate__()
        _state = self.__booster.get_state()
        state['tree_array'], state['threshold_array'], state['leaf_array'] = _state
        return state

    def __setattr__(self, name, value):
        # If the property being set is also a hyperparameter, update it in the _booster object
        # if name in HP_NAMES:
        #     # self.booster_.params = {name: value}
        # elif name == "correlated":
        #     del self.booster_
        # elif name == "shape" and self.shape is not None:
            # raise TypeError(f"{self.__class__.__name__}.shape is immutable")
        super().__setattr__(name, value)

    def fit(self, X, y):
        self.booster_ = (GBDTMulti if self.correlated else GBDTSingle)(self.shape, self.__dict__)
        self.booster_.set_data_regression(X, y)
        self.booster_.train(self.max_iter)
        self.n_trees_ = self.booster_._lib_GetNTrees()  # hacky

    def predict(self, X):
        check_is_fitted(self, "booster_");
        return self.booster_.predict(X)
