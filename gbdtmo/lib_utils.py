import numpy as np
import numpy.ctypeslib as npct
from ctypes import *

array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS')
array_2d_double = npct.ndpointer(dtype=np.double, ndim=2, flags='C_CONTIGUOUS')
array_1d_int = npct.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS')
array_2d_int = npct.ndpointer(dtype=np.int32, ndim=2, flags='C_CONTIGUOUS')
array_1d_uint16 = npct.ndpointer(dtype=np.uint16, ndim=1, flags='C_CONTIGUOUS')
array_2d_uint16 = npct.ndpointer(dtype=np.uint16, ndim=2, flags='C_CONTIGUOUS')


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
        ("max_bins", c_int),
    ]

    def __iter__(self):
        """Iterate through fields of self. This allows calls like `dict(hp)`"""
        for field in self._fields_:
            yield (field[0], getattr(self, field[0]))


def load_lib(path):
    """ Load GBDTMO library from path, and set the API types """
    lib = npct.load_library(path, '.')

    def _s(fun, argtypes, restype=None):
        fun.argtypes = argtypes
        fun.restype = restype

    _s(lib.SetGH, [c_void_p, array_2d_double, array_2d_double])
    _s(lib.Boost, [c_void_p])
    _s(lib.Train, [c_void_p, c_int])
    _s(lib.Dump, [c_void_p, c_char_p])
    _s(lib.Load, [c_void_p, c_char_p])
    _s(lib.Reset, [c_void_p])
    _s(lib.SetTrainData, [c_void_p, array_2d_double, array_2d_double, c_int])
    _s(lib.SetEvalData, [c_void_p, array_2d_double, array_2d_double, c_int])
    _s(lib.SetTrainLabelDouble, [c_void_p, array_2d_double])
    _s(lib.SetTrainLabelInt, [c_void_p, array_2d_int])
    _s(lib.SetEvalLabelDouble, [c_void_p, array_2d_double])
    _s(lib.SetEvalLabelInt, [c_void_p, array_2d_int])
    _s(lib.CalcTrainMaps, [c_void_p])
    _s(lib.Predict, [c_void_p, array_2d_double, array_2d_double, c_int, c_int])
    _s(lib.SingleNew, [HyperParameters], c_void_p)
    _s(lib.MultiNew, [HyperParameters], c_void_p)
    _s(lib.DefaultHyperParameters, None, HyperParameters)

    return lib
