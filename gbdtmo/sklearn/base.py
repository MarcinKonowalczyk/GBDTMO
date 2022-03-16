from ..gbdtmo import GBDTMulti, GBDTSingle, Loss, HyperParameters

from sklearn.base import BaseEstimator, RegressorMixin

HP_NAMES = list(zip(*HyperParameters._fields_))[0]

class GBDTMO(BaseEstimator, RegressorMixin):
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
        params = dict(
            loss=Loss.mse,
            max_depth=max_depth,
            max_leaves=max_leaves,
            seed=seed,
            min_samples=min_samples,
            learning_rate=learning_rate,
            reg_l1=reg_l1,
            reg_l2=reg_l2,
            gamma=gamma,
            early_stop=early_stop,
            verbose=verbose,
            max_caches=max_caches,
            topk=topk,
            one_side=one_side,
            max_bins=max_bins,
        )
        # print(set(params))
        # print(set(HP_NAMES))
        # print(set(HP_NAMES).difference(set(params)))
        # assert set(params) == set(HP_NAMES)
        self.__dict__.update(params)
        self._booster = (GBDTMulti if self.correlated else GBDTSingle)(shape, params)
        
    def __getstate__(self):
        state = super().__getstate__()
        del state['_booster']
        _state = self._booster.get_state()
        state['tree_array'], state['threshold_array'], state['leaf_array'] = _state
        return state

    def __setattr__(self, name, value):
        # If the property being set is also a hyperparameter, update it in the _booster object
        if name in HP_NAMES:
            self._booster.params = {name: value}
        super().__setattr__(name, value)

    def set_eval(self, X, y):
        self._booster.set_eval_data(X, y)

    def fit(self, X, y):
        self._booster.reset()
        self._booster.set_train_data(X, y)
        self._booster.calc_train_maps()
        self._booster.train(self.max_iter)

    def predict(self, X):
        return self._booster.predict(X)