import numpy as np
import numpy.ctypeslib as npct
from ctypes import *

#=========================================================================
#                                                                         
#  #####  ##     ##  ##   ##  ###    ###                                
#  ##     ####   ##  ##   ##  ## #  # ##                                
#  #####  ##  ## ##  ##   ##  ##  ##  ##                                
#  ##     ##    ###  ##   ##  ##      ##                                
#  #####  ##     ##   #####   ##      ##                                
#                                                                         
#=========================================================================

class EnumerationMeta(type(c_uint)):
    def __new__(metacls, name, bases, class_dict):
        cls = type(c_uint).__new__(metacls, name, bases, class_dict)
        if "_members_" in class_dict:
            for key,value in cls._members_:
                setattr(cls, key, cls(value))
        return cls

class Enumeration(c_uint, metaclass=EnumerationMeta):
    def __repr__(self):
        _vk_map = {v:k for k,v in self._members_}
        return f"<{self.__class__.__name__}.{_vk_map[self.value]}: {self.value}>"

#=======================================================================================
#                                                                                       
#    ###    #####    #####      ###    ##    ##   ####                                
#   ## ##   ##  ##   ##  ##    ## ##    ##  ##   ##                                   
#  ##   ##  #####    #####    ##   ##    ####     ###                                 
#  #######  ##  ##   ##  ##   #######     ##        ##                                
#  ##   ##  ##   ##  ##   ##  ##   ##     ##     ####                                 
#                                                                                       
#=======================================================================================

array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS')
array_2d_double = npct.ndpointer(dtype=np.double, ndim=2, flags='C_CONTIGUOUS')
array_1d_int = npct.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS')
array_2d_int = npct.ndpointer(dtype=np.int32, ndim=2, flags='C_CONTIGUOUS')
array_1d_uint16 = npct.ndpointer(dtype=np.uint16, ndim=1, flags='C_CONTIGUOUS')
array_2d_uint16 = npct.ndpointer(dtype=np.uint16, ndim=2, flags='C_CONTIGUOUS')

class Loss(Enumeration):
    _members_ = [
        ("mse", 0),
        ("ce", 1),
        ("ce_column", 2),
        ("bce", 3)
    ]

class HyperParameters(Structure):
    _fields_ = [
        ("inp_dim", c_int),
        ("out_dim", c_int),
        ("loss", Loss),
        ("max_depth", c_int),
        ("max_leaves", c_int),
        ("seed", c_int),
        ("min_samples", c_int),
        ("learning_rate", c_double),
        ("reg_l1", c_double),
        ("reg_l2", c_double),
        ("gamma", c_double),
        ("early_stop", c_int),
        ("verbose", c_bool),
        ("max_caches", c_int),
        ("topk", c_int),
        ("one_side", c_bool),
        ("max_bins", c_int),
        ("alpha", c_double),
        ("eval_fraction", c_double),
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

    _s(lib.Boost, [c_void_p])
    _s(lib.Train, [c_void_p, c_int])
    _s(lib.Reset, [c_void_p])
    _s(lib.SetTrainData, [c_void_p, array_2d_double, array_2d_double, c_int])
    _s(lib.SetEvalData, [c_void_p, array_2d_double, array_2d_double, c_int])
    _s(lib.SetTrainLabelDouble, [c_void_p, array_2d_double])
    _s(lib.SetTrainLabelInt, [c_void_p, array_2d_int])
    _s(lib.SetEvalLabelDouble, [c_void_p, array_2d_double])
    _s(lib.SetEvalLabelInt, [c_void_p, array_2d_int])

    _s(lib.GetDefaultParameters, None, HyperParameters)
    _s(lib.GetCurrentParameters, [c_void_p], HyperParameters)
    _s(lib.SetParameters, [c_void_p, HyperParameters])

    _s(lib.CalcTrainMaps, [c_void_p])
    _s(lib.Predict, [c_void_p, array_2d_double, array_2d_double, c_int, c_int])
    _s(lib.SingleNew, [HyperParameters], c_void_p)
    _s(lib.MultiNew, [HyperParameters], c_void_p)
    _s(lib.Delete, [c_void_p])

    # Functions to get the state of the booster
    _s(lib.GetNTrees, [c_void_p], c_uint)
    _s(lib.GetNonleafSizes, [c_void_p, array_1d_uint16])
    _s(lib.GetLeafSizes, [c_void_p, array_1d_uint16])
    _s(lib.GetNonleafNodes, [c_void_p, array_2d_int, array_1d_double])
    _s(lib.GetLeafNodes, [c_void_p, array_2d_double])

    # _s(lib.GetState, [c_void_p])
    _s(lib.Dump, [c_void_p, c_char_p])
    _s(lib.Load, [c_void_p, c_char_p])

    return lib


import os

LIB = os.path.realpath(os.path.join(os.path.dirname(__file__), "lib.so"))
