import numpy as np
import numpy.ctypeslib as npct
import ctypes
from ctypes import *

array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
array_2d_double = npct.ndpointer(dtype=np.double, ndim=2, flags='CONTIGUOUS')
array_1d_int = npct.ndpointer(dtype=np.int32, ndim=1, flags='CONTIGUOUS')
array_2d_int = npct.ndpointer(dtype=np.int32, ndim=2, flags='CONTIGUOUS')
array_1d_uint16 = npct.ndpointer(dtype=np.uint16, ndim=1, flags='CONTIGUOUS')
array_2d_uint16 = npct.ndpointer(dtype=np.uint16, ndim=2, flags='CONTIGUOUS')


def load_lib(path):
    lib = npct.load_library(path, '.')

    lib.SetBin.argtypes = [c_void_p, array_1d_uint16, array_1d_double]
    lib.SetBin.restype = None
    lib.SetGH.argtypes = [c_void_p, array_2d_double, array_2d_double]
    lib.SetGH.restype = None
    lib.Boost.argtypes = [c_void_p]
    lib.Boost.restype = None
    lib.Train.argtypes = [c_void_p, c_int]
    lib.Train.restype = None
    lib.Dump.argtypes = [c_void_p, c_char_p]
    lib.Dump.restype = None
    lib.Load.argtypes = [c_void_p, c_char_p]
    lib.Load.restype = None

    # Default to 1d array, but this might change in the _set_label call
    lib.SetTrainData.argtypes = [c_void_p, array_2d_uint16, array_2d_double, array_2d_double, c_int]
    lib.SetTrainData.restype = None
    lib.SetEvalData.argtypes = [c_void_p, array_2d_uint16, array_2d_double, array_2d_double, c_int]
    lib.SetEvalData.restype = None
    lib.SetLabelDouble.argtypes = [c_void_p, array_1d_double, c_bool]
    lib.SetLabelDouble.restype = None
    lib.SetLabelInt.argtypes = [c_void_p, array_1d_int, c_bool]
    lib.SetLabelInt.restype = None
    lib.Predict.argtypes = [c_void_p, array_2d_double, array_1d_double, c_int, c_int]
    lib.Predict.restype = None
    lib.Predict2.argtypes = [c_void_p, array_2d_double, array_1d_double, c_int, c_int, c_int]
    lib.Predict2.restype = None
    lib.PredictMulti.argtypes = [c_void_p, array_2d_double, array_1d_double, c_int, c_int, c_int]
    # lib.PredictMulti.argtypes = [c_void_p, array_2d_double, array_2d_double, c_int, c_int, c_int]
    lib.PredictMulti.restype = None

    # lib.Predict.argtypes = [c_void_p, array_2d_double, array_2d_double, c_int, c_int]

    lib.SingleNew.argtypes = [
        c_int, c_char_p, c_int, c_int, c_int, c_int, c_double, c_double, c_double, c_double, c_double, c_int, c_bool,
        c_int
    ]
    lib.SingleNew.restype = c_void_p

    lib.MultiNew.argtypes = [
        c_int, c_int, c_int, c_char_p, c_int, c_int, c_int, c_int, c_double, c_double, c_double, c_double, c_double,
        c_int, c_bool, c_bool, c_int
    ]
    lib.MultiNew.restype = c_void_p

    lib.TrainMulti.argtypes = [c_void_p, c_int, c_int]
    lib.TrainMulti.restype = None
    lib.Reset.argtypes = [c_void_p]
    lib.Reset.restype = None

    return lib


def set_Nth_argtype(lib_fun, N, value):
    """
    Set the argtype of the N'th argument to a C-library function.
    This has to be done this way as opposed to lib.fun.argtypes[N] = value
    since ctypes has custom setter/getter which does not support indexing.
    """
    argtypes = lib_fun.argtypes
    argtypes[N] = value
    lib_fun.argtypes = argtypes


def default_params():
    return dict(
        topk=0,
        loss=b"mse",
        max_depth=4,
        max_leaves=32,
        seed=0,
        min_samples=20,
        lr=0.2,
        reg_l1=0.0,
        reg_l2=1.0,
        gamma=1e-3,
        base_score=0.0,
        early_stop=0,
        one_side=True,
        verbose=True,
        hist_cache=16,
        max_bins=32,
        subsample=1.0,
    )
