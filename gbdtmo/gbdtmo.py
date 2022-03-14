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


class GBDTBase:

    _lib_init = None

    def __init__(self, lib, shape, params={}):

        # Make sure the required parameters are set in the children classes
        for required_attr in ('_lib_init', ):
            if getattr(self, required_attr) is None:
                raise NotImplementedError(f"Attribute '{required_attr}' not set")

        self.lib = lib
        self.inp_dim, self.out_dim = shape

        self.params = dict(self.lib.DefaultHyperParameters())
        self.params.update(dict(inp_dim=self.inp_dim, out_dim=self.out_dim))
        self.params.update(params)
        self.params['loss'] = self._ensure_bytes(self.params['loss'])

        # self.__dict__.update(self.params)

        # Get pointer to the C object
        lib_init = getattr(self.lib, self._lib_init)
        self._booster = lib_init(HyperParameters(**self.params))

    def __del__(self):
        # print("Debug test print on delete. This should free the memory of the C object");
        self.lib.Delete(self._booster)

    @staticmethod
    def _ensure_bytes(string):
        if isinstance(string, str):
            return string.encode()
        elif isinstance(string, bytes):
            return string
        else:
            raise TypeError(
                f"Strings passed to C must be byte arrays. Type '{type(string)}' is not convertible to a byte array.")

    @staticmethod
    def _check_label(y: np.ndarray):
        is_float = (y.dtype == np.float64)
        if not (is_float or y.dtype == np.int32):
            raise TypeError(f"label must be float64 or int32 (not {y.dtype})")
        return is_float

    def _set_train_label(self, y: np.ndarray):
        is_float = self._check_label(y)
        _f = self.lib.SetTrainLabelDouble if is_float else self.lib.SetTrainLabelInt
        _f(self._booster, y)

    def _set_eval_label(self, y: np.ndarray):
        is_float = self._check_label(y)
        _f = self.lib.SetEvalLabelDouble if is_float else self.lib.SetEvalLabelInt
        _f(self._booster, y)

    def boost(self):
        self.lib.Boost(self._booster)

    def dump(self, path):
        path = self._ensure_bytes(path)
        self.lib.Dump(self._booster, path)

    def load(self, path):
        path = self._ensure_bytes(path)
        self.lib.Load(self._booster, path)

    def train(self, num):
        self.lib.Train(self._booster, num)

    def reset(self):
        self.lib.Reset(self._booster)

    def _get_n_trees(self):
        return self.lib.GetNTrees(self._booster)

    def get_state(self):
        N = self._get_n_trees()
        tree_array, threshold_array = self._get_nonleaf_nodes(N)
        leaf_array = self._get_leaf_nodes(N)
        return tree_array, threshold_array, leaf_array

    def _get_nonleaf_sizes(self, n_trees=None):
        """Get the array describing the sizes of each of the trees in the nonleaf array"""
        n_trees = self._get_n_trees() if not n_trees else n_trees
        nonleaf_sizes = np.zeros(n_trees, dtype=np.uint16)
        self.lib.GetNonleafSizes(self._booster, nonleaf_sizes)
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
        """Get the tree array and the corresponding threshold array"""
        tree_sizes = self._get_nonleaf_sizes(n_trees)
        N_nonleaf = np.sum(tree_sizes)
        tree_array = np.full((N_nonleaf, 5), 0, dtype=np.int32)
        threshold_array = np.full(N_nonleaf, 0, dtype=np.double)
        self.lib.GetNonleafNodes(self._booster, tree_array, threshold_array)

        threshold_array = np.squeeze(self._splitter_combiner(threshold_array, tree_sizes))
        tree_array = self._splitter_combiner(tree_array, tree_sizes, pad_value=0)
        
        return tree_array, threshold_array

    def _get_leaf_sizes(self, n_trees=None):
        """Get the array describing the sizes of each of the trees in the leaf array"""
        n_trees = self._get_n_trees() if not n_trees else n_trees
        leaf_sizes = np.zeros(n_trees, dtype=np.uint16)
        self.lib.GetLeafSizes(self._booster, leaf_sizes)
        return leaf_sizes

    def _get_leaf_nodes(self, n_trees=None, _out_dim=None):
        """Get the array describing the values at the leaves"""
        _out_dim = self.out_dim if not _out_dim else _out_dim
        leaf_sizes = self._get_leaf_sizes(n_trees)
        N_leaf = np.sum(leaf_sizes)
        leaf_array = np.full((N_leaf, _out_dim), 0, dtype=np.double)
        self.lib.GetLeafNodes(self._booster, leaf_array)
        return self._splitter_combiner(leaf_array, leaf_sizes)

    def predict(self, X, num_trees=0):
        """ """
        preds = np.full((len(X), self.out_dim), self.params['base_score'], dtype=np.float64)
        self.lib.Predict(self._booster, X, preds, len(X), num_trees)
        return preds

    @staticmethod
    def _check_data(X):
        if not X.dtype == np.float64:
            raise TypeError(f"X must be a float64 (not {X.dtype})")
        raise NotImplementedError("work in progress")

    def set_train_data(self, data, label=None):
        """ """
        self.data_train = np.ascontiguousarray(data)
        self.preds_train = np.full((len(self.data_train), self.out_dim), self.params['base_score'], dtype=np.float64)
        self.lib.SetTrainData(self._booster, self.data_train, self.preds_train, len(self.data_train))

        if label is not None:
            self.label = np.ascontiguousarray(label)
            self._set_train_label(self.label)

    def set_eval_data(self, data, label=None):
        """ """
        self.data_eval = np.ascontiguousarray(data)
        self.preds_eval = np.full((len(self.data_eval), self.out_dim), self.params['base_score'], dtype=np.float64)
        self.lib.SetEvalData(self._booster, self.data_eval, self.preds_eval, len(self.data_eval))

        if label is not None:
            self.label_eval = np.ascontiguousarray(label)
            self._set_eval_label(self.label_eval)

    def calc_train_maps(self):
        self.lib.CalcTrainMaps(self._booster)


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

    _lib_init = "SingleNew"

    # TODO: think about the array ordering

    def set_train_data(self, data, label=None):
        if label is not None: label = label.transpose()
        return super().set_train_data(data, label)

    def set_eval_data(self, data, label=None):
        if label is not None: label = label.transpose()
        return super().set_eval_data(data, label)

    @staticmethod
    def transpose_memory(array):
        return np.transpose(np.reshape(array, (array.shape[1], array.shape[0])))

    def train(self, num):
        super().train(num)
        self.preds_train = self.transpose_memory(self.preds_train)
        self.preds_eval = self.transpose_memory(self.preds_eval)

    def predict(self, X, num_trees=0):
        preds = super().predict(X, num_trees)
        return self.transpose_memory(preds)

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

    _lib_init = "MultiNew"

    def _get_leaf_nodes(self, n_trees=None):
        return super()._get_leaf_nodes(n_trees, _out_dim=None)
