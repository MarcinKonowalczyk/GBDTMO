import numpy as np
from .histogram import get_bins_maps
from .lib_utils import *

#======================================================================
#
#  ##   ##  ######  ##  ##       ####
#  ##   ##    ##    ##  ##      ##
#  ##   ##    ##    ##  ##       ###
#  ##   ##    ##    ##  ##         ##
#   #####     ##    ##  ######  ####
#
#======================================================================


class GBDTBase:

    _lib_init = None

    def __init__(self, lib, shape, params={}, max_bins=32):

        # Make sure the required parameters are set in the children classes
        for required_attr in ('_lib_init', ):
            if getattr(self, required_attr) is None:
                raise NotImplementedError(f"Attribute '{required_attr}' not set")

        self.lib = lib
        self.inp_dim, self.out_dim = shape
        self.max_bins = max_bins

        hp = self.lib.DefaultHyperParameters()
        hp.inp_dim = self.inp_dim
        hp.out_dim = self.out_dim
        self.params = dict(hp)

        # Make sure loss is a bytes string, to make the api a bit more user-friendly
        params['loss'] = params['loss'].encode() if isinstance(params['loss'], str) else params['loss']

        self.params.update(params)
        # self.__dict__.update(self.params)

        # NOTE: The values in DEFAULT_PARAMS are in the correct order for the library call
        lib_init = getattr(self.lib, self._lib_init)
        self._boostnode = lib_init(HyperParameters(**self.params))

    def _set_bin(self, bins):
        num = np.fromiter((len(b) for b in bins), dtype=np.uint16)
        value = np.concatenate(bins, axis=0)
        self.lib.SetBin(self._boostnode, num, value)

    def _set_label(self, x: np.array, is_train: bool):
        Nd = x.ndim
        if not ((is1d := (Nd == 1)) or Nd == 2):
            raise TypeError(f"label must be 1D or 2D array (not {Nd})")
        dtype = x.dtype
        if not ((isfloat := (dtype == np.float64)) or dtype == np.int32):
            raise TypeError(f"label must be float64 or int32 (not {dtype})")

        if isfloat:
            set_Nth_argtype(self.lib.SetLabelDouble, 1, array_1d_double if is1d else array_2d_double)
            self.lib.SetLabelDouble(self._boostnode, x, is_train)
        else:
            set_Nth_argtype(self.lib.SetLabelDouble, 1, array_1d_int if is1d else array_2d_int)
            self.lib.SetLabelInt(self._boostnode, x, is_train)

    def boost(self):
        self.lib.Boost(self._boostnode)

    def dump(self, path):
        self.lib.Dump(self._boostnode, path)

    def load(self, path):
        self.lib.Load(self._boostnode, path)

    def train(self, num):
        self.lib.Train(self._boostnode, num)

    def reset(self):
        self.lib.Reset(self._boostnode)


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

    def set_data(self, train_set: tuple = None, eval_set: tuple = None):

        if train_set is not None:
            self.data = np.ascontiguousarray(train_set[0])
            self.label = np.ascontiguousarray(train_set[1].transpose())
            self.bins, self.maps = get_bins_maps(self.data, self.max_bins)
            self._set_bin(self.bins)
            self.maps = np.ascontiguousarray(self.maps.transpose())
            self.preds_train = np.full(len(self.data) * self.out_dim, self.params['base_score'], dtype=np.float64)
            set_Nth_argtype(self.lib.SetTrainData, 3, array_1d_double)
            self.lib.SetTrainData(self._boostnode, self.maps, self.data, self.preds_train, len(self.data))
            if self.label is not None:
                self._set_label(self.label, True)

        if eval_set is not None:
            self.data_eval = np.ascontiguousarray(eval_set[0])
            self.label_eval = np.ascontiguousarray(eval_set[1].transpose())
            self.preds_eval = np.full(len(self.data_eval) * self.out_dim, self.params['base_score'], dtype=np.float64)
            maps = np.zeros((1, 1), dtype=np.uint16)
            set_Nth_argtype(self.lib.SetEvalData, 3, array_1d_double)
            self.lib.SetEvalData(self._boostnode, maps, self.data_eval, self.preds_eval, len(self.data_eval))
            if self.label_eval is not None:
                self._set_label(self.label_eval, False)

    def predict(self, X, num_trees=0):
        N = X.shape[0]
        preds = np.full(N * self.out_dim, self.params['base_score'], dtype=np.float64)
        set_Nth_argtype(self.lib.Predict, 2, array_1d_double)
        self.lib.Predict(self._boostnode, X, preds, N, num_trees)
        return np.transpose(np.reshape(preds, (self.out_dim, N)))


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

    def set_data(self, train_set: tuple = None, eval_set: tuple = None):

        if train_set is not None:
            self.data, self.label = map(np.ascontiguousarray, train_set)
            self.bins, self.maps = get_bins_maps(self.data, self.max_bins)
            self._set_bin(self.bins)
            self.maps = np.ascontiguousarray(self.maps.transpose())
            self.preds_train = np.full((len(self.data), self.out_dim), self.params['base_score'], dtype=np.float64)
            set_Nth_argtype(self.lib.SetTrainData, 3, array_2d_double)
            self.lib.SetTrainData(self._boostnode, self.maps, self.data, self.preds_train, len(self.data))
            if self.label is not None:
                self._set_label(self.label, True)

        if eval_set is not None:
            self.data_eval, self.label_eval = map(np.ascontiguousarray, eval_set)
            self.preds_eval = np.full((len(self.data_eval), self.out_dim), self.params['base_score'], dtype=np.float64)
            maps = np.zeros((1, 1), dtype=np.uint16)
            set_Nth_argtype(self.lib.SetEvalData, 3, array_2d_double)
            self.lib.SetEvalData(self._boostnode, maps, self.data_eval, self.preds_eval, len(self.data_eval))
            if self.label_eval is not None:
                self._set_label(self.label_eval, False)

    def predict(self, X, num_trees=0):
        preds = np.full((len(X), self.out_dim), self.params['base_score'], dtype=np.float64)
        set_Nth_argtype(self.lib.Predict, 2, array_2d_double)
        self.lib.Predict(self._boostnode, X, preds, len(X), num_trees)
        return preds
