import numpy as np
from .lib_utils import *

#=================================================================
#
#  #####     ###     ####  #####
#  ##  ##   ## ##   ##     ##
#  #####   ##   ##   ###   #####
#  ##  ##  #######     ##  ##
#  #####   ##   ##  ####   #####
#
#=================================================================

BASE_SCORE = 0.0


class GBDTBase(BoosterLibWrapper):
    def __init__(self, shape, params={}):
        super().__init__()
        self.inp_dim, self.out_dim = shape
        _params = dict(self._lib.GetDefaultParameters())
        _params.update(params)

        lib_init = getattr(self._lib, self._lib_init_name)
        self._booster = lib_init(Shape(*shape), HyperParameters(**_params))

    @property
    def params(self):
        return dict(self._lib_GetCurrentParameters())

    @params.setter
    def params(self, value):
        if isinstance(value, HyperParameters):
            pass
        elif isinstance(value, dict):
            # Update the existing params with the suplied dict
            _params = self.params
            for key in value:
                if key not in _params:
                    raise KeyError(f"Unknown key '{key}'")
            _params.update(value)
            value = HyperParameters(**_params)
        else:
            raise TypeError(f"'value' must be of type 'HyperParameters' or 'dict', not '{type(value)}'")
        self._lib_SetParameters(value)

    def dump(self, path):
        self._lib_Dump(path)

    def load(self, path):
        self._lib_Load(path)

    def get_state(self):
        N = self._lib_GetNTrees()
        tree_array, threshold_array = self._get_nonleaf_nodes(N)
        leaf_array = self._get_leaf_nodes(N)
        return tree_array, threshold_array, leaf_array

    def _get_nonleaf_sizes(self, n_trees=None):
        """ Get the array describing the sizes of each of the trees in the nonleaf array """
        n_trees = self._lib_GetNTrees() if not n_trees else n_trees
        nonleaf_sizes = np.zeros(n_trees, dtype=np.uint16)
        self._lib_GetNonleafSizes(nonleaf_sizes)
        return nonleaf_sizes

    @staticmethod
    def _splitter_combiner(X, sizes, pad_value=np.nan):
        """ Split array according to sizes, pad with value and recombine into a 3d array """
        if len(X.shape) == 1: X = X.reshape(-1, 1)  # If array is 1d, add a dimension
        splits = np.cumsum(sizes)
        assert splits[-1] == X.shape[0]
        X = np.split(X, splits[:-1])
        max_size = np.max(sizes)
        padder = lambda x: np.pad(x, ((0, max_size - len(x)), (0, 0)), mode="constant", constant_values=pad_value)
        return np.ascontiguousarray(np.moveaxis(np.stack([padder(x) for x in X], axis=0), 1, 2))

    def _get_nonleaf_nodes(self, n_trees=None):
        """ Get the tree array and the corresponding threshold array """
        tree_sizes = self._get_nonleaf_sizes(n_trees)
        N_nonleaf = np.sum(tree_sizes)
        tree_array = np.full((N_nonleaf, 5), 0, dtype=np.int32)
        threshold_array = np.full(N_nonleaf, 0, dtype=np.double)
        self._lib_GetNonleafNodes(tree_array, threshold_array)

        if n_trees > 0:
            threshold_array = np.squeeze(self._splitter_combiner(threshold_array, tree_sizes))
            tree_array = self._splitter_combiner(tree_array, tree_sizes, pad_value=0)

        return tree_array, threshold_array

    def _get_leaf_sizes(self, n_trees=None):
        """Get the array describing the sizes of each of the trees in the leaf array"""
        n_trees = self._lib_GetNTrees() if not n_trees else n_trees
        leaf_sizes = np.zeros(n_trees, dtype=np.uint16)
        self._lib_GetLeafSizes(leaf_sizes)
        return leaf_sizes

    def _get_leaf_nodes(self, n_trees=None, _out_dim=None):
        """Get the array describing the values at the leaves"""
        _out_dim = self.out_dim if not _out_dim else _out_dim
        leaf_sizes = self._get_leaf_sizes(n_trees)
        N_leaf = np.sum(leaf_sizes)
        leaf_array = np.full((N_leaf, _out_dim), 0, dtype=np.double)
        self._lib_GetLeafNodes(leaf_array)
        if n_trees > 0:
            leaf_array = self._splitter_combiner(leaf_array, leaf_sizes)
        return leaf_array

    def predict(self, X, num_trees=0):
        """ """
        preds = np.full((len(X), self.out_dim), BASE_SCORE, dtype=np.float64)
        self._lib_Predict(X, preds, len(X), num_trees)
        return preds

    @staticmethod
    def _check_data(X: np.ndarray):
        if not X.dtype == np.float64:
            raise TypeError(f"X must be a float64 (not {X.dtype})")
        return np.ascontiguousarray(X)

    @staticmethod
    def _check_label(y: np.ndarray):
        if not (y.dtype == np.float64) or (y.dtype == np.int32):
            raise TypeError(f"label must be float64 or int32 (not {y.dtype})")
        return np.ascontiguousarray(y)
        
    def set_data_regression(self, X, y):
        """ """
        self._X, self._y = self._check_data(X), self._check_label(y)
        self._yp = np.full((len(self._X), self.out_dim), BASE_SCORE, dtype=np.float64)
        self._lib_SetDataRegression(self._X, self._yp, self._y, len(self._X))
        self._lib_Calc()

    def set_data_classification(self, X, y):
        """ """
        raise NotImplementedError
        # self._lib_SetDataClassification(self._X, self._yp, self._y, len(self.data_train))

    def train(self, num):
        self._lib_Train(num)

    # def fit(self, X, y):
    #     self._set_data_regression(X, y)
    #     self._lib_Train(num)


#================================================================================
#
#   ####  ##  ##     ##   ####    ##      #####
#  ##     ##  ####   ##  ##       ##      ##
#   ###   ##  ##  ## ##  ##  ###  ##      #####
#     ##  ##  ##    ###  ##   ##  ##      ##
#  ####   ##  ##     ##   ####    ######  #####
#
#================================================================================


class GBDTSingle(GBDTBase):

    _lib_init_name = "SingleNew"

    # TODO: think about the array ordering

    def set_data_regression(self, X, y):
        return super().set_data_regression(X, y.transpose())

    @staticmethod
    def transpose_memory(array):
        return np.transpose(np.reshape(array, (array.shape[1], array.shape[0])))

    def train(self, num):
        super().train(num)
        self._yp = self.transpose_memory(self._yp)

    def predict(self, X, num_trees=0):
        yp = super().predict(X, num_trees)
        return self.transpose_memory(yp)

    def _get_nonleaf_nodes(self, n_trees=None):
        """Get the array describing the tree structure and the corresponding thresholds"""
        tree_array, threshold_array = super()._get_nonleaf_nodes(n_trees)
        _reshape = (n_trees // self.out_dim, self.out_dim)
        tree_array = tree_array.reshape(*_reshape, *tree_array.shape[1:])
        threshold_array = threshold_array.reshape(*_reshape, *threshold_array.shape[1:])
        return tree_array, threshold_array

    def _get_leaf_nodes(self, n_trees=None):
        """Get the array describing the values at the leaves"""
        leaf_array = super()._get_leaf_nodes(n_trees, _out_dim=1)
        leaf_array = leaf_array.reshape(n_trees // self.out_dim, self.out_dim, leaf_array.shape[2])
        return leaf_array


#===========================================================================
#
#  ###    ###  ##   ##  ##      ######  ##
#  ## #  # ##  ##   ##  ##        ##    ##
#  ##  ##  ##  ##   ##  ##        ##    ##
#  ##      ##  ##   ##  ##        ##    ##
#  ##      ##   #####   ######    ##    ##
#
#===========================================================================


class GBDTMulti(GBDTBase):

    _lib_init_name = "MultiNew"

    def _get_leaf_nodes(self, n_trees=None):
        return super()._get_leaf_nodes(n_trees, _out_dim=None)
