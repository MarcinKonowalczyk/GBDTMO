import numpy as np
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

        # Make sure loss is a bytes string, to make the api a bit more user-friendly
        if isinstance(self.params['loss'], str):
            self.params['loss'] = self.params['loss'].encode()

        # self.__dict__.update(self.params)

        # NOTE: The values in DEFAULT_PARAMS are in the correct order for the library call
        lib_init = getattr(self.lib, self._lib_init)
        self._boostnode = lib_init(HyperParameters(**self.params))

    def _set_label(self, y: np.array, is_train: bool):
        dtype = y.dtype
        if not ((is_float := (dtype == np.float64)) or dtype == np.int32):
            raise TypeError(f"label must be float64 or int32 (not {dtype})")

        if is_float and is_train:
            self.lib.SetTrainLabelDouble(self._boostnode, y)
        elif is_float and not is_train:
            self.lib.SetEvalLabelDouble(self._boostnode, y)
        elif not is_float and is_train:
            self.lib.SetTrainLabelInt(self._boostnode, y)
        else:  # not is_float and not is_train
            self.lib.SetEvalLabelInt(self._boostnode, y)

        # if is_float:
        #     if is_train:
        #         self.lib.SetTrainLabelDouble(self._boostnode, y)
        #     else:
        #         self.lib.SetEvalLabelDouble(self._boostnode, y)
        # else:
        #     if is_train:
        #         self.lib.SetTrainLabelInt(self._boostnode, y)
        #     else:
        #         self.lib.SetEvalLabelInt(self._boostnode, y)

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

    def predict(self, X, num_trees=0):
        preds = np.full((len(X), self.out_dim), self.params['base_score'], dtype=np.float64)
        self.lib.Predict(self._boostnode, X, preds, len(X), num_trees)
        return preds

    def set_train_data(self, data, label=None):
        """ """
        self.data = np.ascontiguousarray(data)
        self.preds_train = np.full((len(self.data), self.out_dim), self.params['base_score'], dtype=np.float64)
        self.lib.SetTrainData(self._boostnode, self.data, self.preds_train, len(self.data))

        if label is not None:
            self.label = np.ascontiguousarray(label)
            self._set_label(self.label, True)

    def set_eval_data(self, data, label=None):
        """ """
        self.data_eval = np.ascontiguousarray(data)
        self.preds_eval = np.full((len(self.data_eval), self.out_dim), self.params['base_score'], dtype=np.float64)
        # maps = np.zeros((1, 1), dtype=np.uint16)  # Eval set does not need maps
        self.lib.SetEvalData(self._boostnode, self.data_eval, self.preds_eval, len(self.data_eval))

        if label is not None:
            self.label_eval = np.ascontiguousarray(label)
            self._set_label(self.label_eval, False)

    def calc_train_maps(self):
        print(f"calc_train_maps")
        self.lib.CalcTrainMaps(self._boostnode)


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
