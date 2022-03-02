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


class BoostUtils:
    def __init__(self, lib):
        self.lib = lib
        self._boostnode = None

    def _set_gh(self, g, h):
        self.lib.SetGH(self._boostnode, g, h)

    def _set_bin(self, bins):
        num, value = [], []
        for i, _ in enumerate(bins):
            num.append(len(_))
        num = np.array(num, np.uint16)
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


#================================================================================
#
#   ####  ##  ##     ##   ####    ##      #####
#  ##     ##  ####   ##  ##       ##      ##
#   ###   ##  ##  ## ##  ##  ###  ##      #####
#     ##  ##  ##    ###  ##   ##  ##      ##
#  ####   ##  ##     ##   ####    ######  #####
#
#================================================================================


class GBDTSingle(BoostUtils):
    def __init__(self, lib, out_dim=1, params={}):
        super().__init__(lib)
        self.out_dim = out_dim
        self.params = default_params()
        # Make sure loss is a bytes string, to make the api a bit more user-friendly
        params['loss'] = params['loss'].encode() if isinstance(params['loss'], str) else params['loss']
        self.params.update(params)
        self.__dict__.update(self.params)

    def set_booster(self, inp_dim):
        print(f"inp_dim = {inp_dim}")
        self._boostnode = self.lib.SingleNew(inp_dim, self.params['loss'], self.params['max_depth'],
                                             self.params['max_leaves'], self.params['seed'], self.params['min_samples'],
                                             self.params['lr'], self.params['reg_l1'], self.params['reg_l2'],
                                             self.params['gamma'], self.params['base_score'], self.params['early_stop'],
                                             self.params['verbose'], self.params['hist_cache'])

    def set_data(self, train_set: tuple = None, eval_set: tuple = None):

        if train_set is not None:
            self.data, self.label = train_set
            self.set_booster(self.data.shape[-1])
            self.bins, self.maps = get_bins_maps(self.data, self.max_bins)
            self._set_bin(self.bins)
            self.maps = np.ascontiguousarray(self.maps.transpose())
            self.preds_train = np.full(len(self.data) * self.out_dim, self.base_score, dtype=np.float64)
            set_Nth_argtype(self.lib.SetTrainData, 3, array_1d_double)
            self.lib.SetTrainData(self._boostnode, self.maps, self.data, self.preds_train, len(self.data))
            if self.label is not None:
                self._set_label(self.label, True)

        if eval_set is not None:
            self.data_eval, self.label_eval = eval_set
            self.preds_eval = np.full(len(self.data_eval) * self.out_dim, self.base_score, dtype=np.float64)
            maps = np.zeros((1, 1), dtype=np.uint16)
            set_Nth_argtype(self.lib.SetEvalData, 3, array_1d_double)
            self.lib.SetEvalData(self._boostnode, maps, self.data_eval, self.preds_eval, len(self.data_eval))
            if self.label_eval is not None:
                self._set_label(self.label_eval, False)

    def train(self, num):
        if self.out_dim == 1:
            self.lib.Train(self._boostnode, num)
        else:
            self.lib.TrainMulti(self._boostnode, num, self.out_dim)

    def predict(self, X, num_trees=0):

        N = X.shape[0]
        preds = np.full(N * self.out_dim, self.base_score, dtype=np.float64)
        if self.out_dim == 1:
            set_Nth_argtype(self.lib.Predict, 2, array_1d_double)
            self.lib.Predict(self._boostnode, X, preds, N, num_trees)
        else:
            print(f"self._boostnode = {self._boostnode}")
            print(f"X.shape = {X.shape}")
            print(f"preds.shape = {preds.shape}")
            print(f"N = {N}")
            print(f"self.out_dim = {self.out_dim}")
            print(f"num_trees = {num_trees}")
            print('about to call lib.Predict2 ...')
            # self.lib.PredictMulti(self._boostnode, X, preds, N, self.out_dim, num_trees)
            self.lib.Predict2(self._boostnode, X, preds, N, self.out_dim, num_trees)
            print('out of the lib.Predict2')
            preds = np.transpose(np.reshape(preds, (self.out_dim, N)))

        return preds

    def reset(self):
        self.lib.Reset(self._boostnode)


#===========================================================================
#
#  ###    ###  ##   ##  ##      ######  ##
#  ## #  # ##  ##   ##  ##        ##    ##
#  ##  ##  ##  ##   ##  ##        ##    ##
#  ##      ##  ##   ##  ##        ##    ##
#  ##      ##   #####   ######    ##    ##
#
#===========================================================================


class GBDTMulti(BoostUtils):
    def __init__(self, lib, out_dim=1, params={}):
        super().__init__(lib)
        self.out_dim = out_dim
        self.params = default_params()
        # Make sure loss is a bytes string, to make the api a bit more user-friendly
        params['loss'] = params['loss'].encode() if isinstance(params['loss'], str) else params['loss']
        self.params.update(params)
        self.__dict__.update(self.params)

    def set_booster(self, inp_dim):
        ret = self.lib.MultiNew(inp_dim, self.out_dim, self.params['topk'], self.params['loss'],
                                self.params['max_depth'], self.params['max_leaves'], self.params['seed'],
                                self.params['min_samples'], self.params['lr'], self.params['reg_l1'],
                                self.params['reg_l2'], self.params['gamma'], self.params['base_score'],
                                self.params['early_stop'], self.params['one_side'], self.params['verbose'],
                                self.params['hist_cache'])
        self._boostnode = ret

    def set_data(self, train_set: tuple = None, eval_set: tuple = None):

        if train_set is not None:
            self.data, self.label = train_set
            self.set_booster(self.data.shape[-1])
            self.bins, self.maps = get_bins_maps(self.data, self.max_bins)
            self._set_bin(self.bins)
            self.maps = np.ascontiguousarray(self.maps.transpose())
            self.preds_train = np.full((len(self.data), self.out_dim), self.base_score, dtype=np.float64)
            set_Nth_argtype(self.lib.SetTrainData, 3, array_2d_double)
            self.lib.SetTrainData(self._boostnode, self.maps, self.data, self.preds_train, len(self.data))
            if self.label is not None:
                self._set_label(self.label, True)

        if eval_set is not None:
            self.data_eval, self.label_eval = eval_set
            self.preds_eval = np.full((len(self.data_eval), self.out_dim), self.base_score, dtype=np.float64)
            maps = np.zeros((1, 1), dtype=np.uint16)
            set_Nth_argtype(self.lib.SetEvalData, 3, array_2d_double)
            self.lib.SetEvalData(self._boostnode, maps, self.data_eval, self.preds_eval, len(self.data_eval))

            if self.label_eval is not None:
                self._set_label(self.label_eval, False)

    def predict(self, x, num_trees=0):
        preds = np.full((len(x), self.out_dim), self.base_score, dtype=np.float64)
        set_Nth_argtype(self.lib.Predict, 2, array_2d_double)
        self.lib.Predict(self._boostnode, x, preds, len(x), num_trees)
        return preds
