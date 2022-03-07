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

class HyperParameters(Structure):
    _fields_ = [
        ("inp_dim", c_int),
        ("out_dim", c_int),
        ("loss", c_char_p),
        ("max_depth", c_int),
        ("max_leaves", c_int),
        ("seed", c_int),
        ("min_samples", c_int),
        ("lr", c_double),
        ("reg_l1", c_double),
        ("reg_l2", c_double),
        ("gamma", c_double),
        ("base_score", c_double),
        ("early_stop", c_int),
        ("verbose", c_bool),
        ("max_caches", c_int),
        ("topk", c_int),
        ("one_side", c_bool),
    ]

    def __iter__(self):
        """Iterate through fields of self. This allows calls like `dict(hp)`"""
        for field in self._fields_:
            yield (field[0], getattr(self, field[0]))


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
    lib.Reset.argtypes = [c_void_p]
    lib.Reset.restype = None

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

    lib.SingleNew.argtypes = [HyperParameters]
    lib.SingleNew.restype = c_void_p
    lib.MultiNew.argtypes = [HyperParameters]
    lib.MultiNew.restype = c_void_p

    lib.DefaultHyperParameters.argtypes = None
    lib.DefaultHyperParameters.restype = HyperParameters

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
