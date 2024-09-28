# Default Library path
import os
from ctypes import *
from typing import Optional, Union

import numpy as np
import numpy.ctypeslib as npct

DEFAULT_LIB_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), "lib.so"))

# =========================================================================
#
#  #####  ##     ##  ##   ##  ###    ###
#  ##     ####   ##  ##   ##  ## #  # ##
#  #####  ##  ## ##  ##   ##  ##  ##  ##
#  ##     ##    ###  ##   ##  ##      ##
#  #####  ##     ##   #####   ##      ##
#
# =========================================================================


class EnumerationMeta(type(c_uint)):
    def __new__(metacls, name, bases, class_dict):
        cls = type(c_uint).__new__(metacls, name, bases, class_dict)
        if "_members_" in class_dict:
            for key, value in cls._members_:
                setattr(cls, key, cls(value))
        return cls


class Enumeration(c_uint, metaclass=EnumerationMeta):
    def __repr__(self):
        _vk_map = {v: k for k, v in self._members_}
        return f"<{self.__class__.__name__}.{_vk_map[self.value]}: {self.value}>"


# ==================================================================================
#
#  ####    ######  ##    ##  #####   #####   ####
#  ##  ##    ##     ##  ##   ##  ##  ##     ##
#  ##  ##    ##      ####    #####   #####   ###
#  ##  ##    ##       ##     ##      ##        ##
#  ####      ##       ##     ##      #####  ####
#
# ==================================================================================

array_1d_float = npct.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
array_2d_float = npct.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS")
array_1d_int = npct.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
array_2d_int = npct.ndpointer(dtype=np.int32, ndim=2, flags="C_CONTIGUOUS")
array_1d_uint16 = npct.ndpointer(dtype=np.uint16, ndim=1, flags="C_CONTIGUOUS")
array_2d_uint16 = npct.ndpointer(dtype=np.uint16, ndim=2, flags="C_CONTIGUOUS")


class Loss(Enumeration):
    _members_ = [("mse", 0), ("ce", 1), ("ce_column", 2), ("bce", 3)]


class IterStructureMixin:
    def __iter__(self):
        """Iterate through fields of self. This allows calls like `dict(hp)`"""
        for field in self._fields_:
            yield (field[0], getattr(self, field[0]))


class HyperParameters(Structure, IterStructureMixin):
    _fields_ = [
        ("loss", Loss),
        ("max_depth", c_uint),
        ("max_leaves", c_uint),
        ("seed", c_uint),
        ("min_samples", c_uint),
        ("learning_rate", c_float),
        ("reg_l1", c_float),
        ("reg_l2", c_float),
        ("gamma", c_float),
        ("early_stop", c_uint),
        ("verbose", c_bool),
        ("max_caches", c_uint),
        ("topk", c_uint),
        ("one_side", c_bool),
        ("max_bins", c_uint),
        ("alpha", c_float),
        ("eval_fraction", c_float),
    ]


class Shape(Structure, IterStructureMixin):
    _fields_ = [
        ("inp_dim", c_size_t),
        ("out_dim", c_size_t),
    ]


class c_BoosterBase_p(c_void_p):
    pass


# ==============================================================================================
#
#  ##       #####     ###    ####          ##      ##  #####
#  ##      ##   ##   ## ##   ##  ##        ##      ##  ##  ##
#  ##      ##   ##  ##   ##  ##  ##        ##      ##  #####
#  ##      ##   ##  #######  ##  ##        ##      ##  ##  ##
#  ######   #####   ##   ##  ####          ######  ##  #####
#
# ==============================================================================================


def load_lib(path: str) -> CDLL:
    """Load GBDTMO library from path, and set the API types"""
    lib = npct.load_library(path, ".")

    def _s(fun, argtypes, restype=None):
        fun.argtypes = argtypes
        fun.restype = restype

    _s(lib.Train, [c_BoosterBase_p, c_int])
    _s(lib.Reset, [c_BoosterBase_p])
    _s(
        lib.SetDataRegression,
        [c_BoosterBase_p, array_2d_float, array_2d_float, array_2d_float, c_int],
    )
    _s(
        lib.SetDataClassification,
        [c_BoosterBase_p, array_2d_float, array_2d_float, array_2d_int, c_int],
    )

    _s(lib.GetDefaultParameters, None, HyperParameters)
    _s(lib.GetCurrentParameters, [c_BoosterBase_p], HyperParameters)
    _s(lib.SetParameters, [c_BoosterBase_p, HyperParameters])

    _s(lib.Calc, [c_BoosterBase_p])
    _s(lib.Predict, [c_BoosterBase_p, array_2d_float, array_2d_float, c_int, c_int])
    _s(lib.SingleNew, [Shape, HyperParameters], c_BoosterBase_p)
    _s(lib.MultiNew, [Shape, HyperParameters], c_BoosterBase_p)
    _s(lib.Delete, [c_BoosterBase_p])

    # Functions to get the state of the booster
    _s(lib.GetNTrees, [c_BoosterBase_p], c_uint)
    _s(lib.GetNonleafSizes, [c_BoosterBase_p, array_1d_uint16])
    _s(lib.GetLeafSizes, [c_BoosterBase_p, array_1d_uint16])
    _s(lib.GetNonleafNodes, [c_BoosterBase_p, array_2d_int, array_1d_float])
    _s(lib.GetLeafNodes, [c_BoosterBase_p, array_2d_float])

    _s(lib.Dump, [c_BoosterBase_p, c_char_p])
    _s(lib.Load, [c_BoosterBase_p, c_char_p])

    return lib


# ================================================================================================
#
#  ##      ##  #####      ###    #####   #####   #####  #####
#  ##      ##  ##  ##    ## ##   ##  ##  ##  ##  ##     ##  ##
#  ##  ##  ##  #####    ##   ##  #####   #####   #####  #####
#  ##  ##  ##  ##  ##   #######  ##      ##      ##     ##  ##
#   ###  ###   ##   ##  ##   ##  ##      ##      #####  ##   ##
#
# ================================================================================================


class BoosterLibWrapper:
    """Wrapper for the GBDTMO shared library"""

    _lib_init_name = None

    def __new__(cls, *args, **kwargs):
        # Make sure the required parameters are set in the children classes
        for required_attr in ("_lib_init_name",):
            if getattr(cls, required_attr) is None:
                raise NotImplementedError(f"Attribute '{required_attr}' not set in the child class")
        return super().__new__(cls)

    def __init__(self, lib: Optional[CDLL] = None) -> None:
        self._lib = lib if lib is not None else load_lib(DEFAULT_LIB_PATH)
        # default_params = dict(self._lib.GetDefaultParameters())
        # lib_init = getattr(self._lib, self._lib_init_name)
        # self._booster = lib_init(HyperParameters(**default_params))

    def __delete__(self) -> None:
        self._lib.Delete(self._booster)

    def _lib_Train(self, n_iter: int) -> None:
        self._lib.Train(self._booster, n_iter)

    def _lib_Reset(self) -> None:
        self._lib.Reset(self._booster)

    def _lib_SetDataRegression(self, X: array_2d_float, yp: array_2d_float, y: array_2d_float, n: int) -> None:
        self._lib.SetDataRegression(self._booster, X, yp, y, n)

    def _lib_SetDataClassification(self, X: array_2d_float, yp: array_2d_float, y: array_2d_int, n: int) -> None:
        self._lib.SetDataClassification(self._booster, X, yp, y, n)

    def _lib_GetDefaultParameters(self) -> HyperParameters:
        return self._lib.GetDefaultParameters()

    def _lib_GetCurrentParameters(self) -> HyperParameters:
        return self._lib.GetCurrentParameters(self._booster)

    def _lib_SetParameters(self, hp: HyperParameters) -> None:
        self._lib.SetParameters(self._booster, hp)

    def _lib_Calc(self) -> None:
        self._lib.Calc(self._booster)

    def _lib_Predict(self, X: array_2d_float, yp: array_2d_float, n: int, num_trees: int) -> None:
        self._lib.Predict(self._booster, X, yp, n, num_trees)

    def _lib_GetNTrees(self) -> int:
        return self._lib.GetNTrees(self._booster)

    def _lib_GetNonleafSizes(self, nonleaf_sizes: array_1d_uint16) -> None:
        self._lib.GetNonleafSizes(self._booster, nonleaf_sizes)

    def _lib_GetLeafSizes(self, leaf_sizes: array_1d_uint16) -> None:
        self._lib.GetLeafSizes(self._booster, leaf_sizes)

    def _lib_GetNonleafNodes(self, trees: array_2d_int, nonleaves: array_1d_float) -> None:
        self._lib.GetNonleafNodes(self._booster, trees, nonleaves)

    def _lib_GetLeafNodes(self, leaves: array_2d_float) -> None:
        self._lib.GetLeafNodes(self._booster, leaves)

    @staticmethod
    def _ensure_bytes(string: Union[str, bytes]) -> bytes:
        if isinstance(string, str):
            return string.encode()
        elif isinstance(string, bytes):
            return string
        else:
            msg = f"Strings passed to C must be byte arrays. Type '{type(string)}' is not convertible to a byte array."
            raise TypeError(msg)

    def _lib_Dump(self, path: str) -> None:
        path = self._ensure_bytes(path)
        self._lib.Dump(self._booster, path)

    def _lib_Load(self, path: str) -> None:
        path = self._ensure_bytes(path)
        self._lib.Load(self._booster, path)
