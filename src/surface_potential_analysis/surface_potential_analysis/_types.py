from __future__ import annotations

from typing import Any, TypeVar, Union

import numpy as np

# The 6 `<X>Like_co` type-aliases below represent all scalars that can be
# coerced into `<X>` (with the casting rule `same_kind`)
_BoolLike_co = Union[bool, np.bool_]
_UIntLike_co = Union[_BoolLike_co, np.unsignedinteger]
_IntLike_co = Union[int, np.integer]
_FloatLike_co = Union[_IntLike_co, float, np.floating]
_ComplexLike_co = Union[_FloatLike_co, complex, np.complexfloating]
_TD64Like_co = Union[_IntLike_co, np.timedelta64]

_S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])

SingleFlatIndexLike = _IntLike_co
ArrayFlatIndexLike = np.ndarray[_S0Inv, np.dtype[np.int_]]
FlatIndexLike = SingleFlatIndexLike | ArrayFlatIndexLike[Any]

SingleStackedIndexLike = tuple[_IntLike_co, ...]
ArrayStackedIndexLike = tuple[np.ndarray[_S0Inv, np.dtype[np.int_]], ...]
StackedIndexLike = SingleStackedIndexLike | ArrayStackedIndexLike[Any]

SingleIndexLike = SingleFlatIndexLike | SingleStackedIndexLike
ArrayIndexLike = ArrayFlatIndexLike[_S0Inv] | ArrayStackedIndexLike[_S0Inv]
IndexLike = SingleIndexLike | ArrayIndexLike[Any]

SingleStackedIndexLike2d = tuple[_IntLike_co, _IntLike_co]
ArrayStackedIndexLike2d = tuple[
    np.ndarray[_S0Inv, np.dtype[np.int_]],
    np.ndarray[_S0Inv, np.dtype[np.int_]],
]
StackedIndexLike2d = SingleStackedIndexLike2d | ArrayStackedIndexLike2d[Any]
IndexLike2d = FlatIndexLike | StackedIndexLike2d

SingleStackedIndexLike3d = tuple[_IntLike_co, _IntLike_co, _IntLike_co]
ArrayStackedIndexLike3d = tuple[
    np.ndarray[_S0Inv, np.dtype[np.int_]],
    np.ndarray[_S0Inv, np.dtype[np.int_]],
    np.ndarray[_S0Inv, np.dtype[np.int_]],
]
StackedIndexLike3d = SingleStackedIndexLike3d | ArrayStackedIndexLike3d[Any]
IndexLike3d = FlatIndexLike | StackedIndexLike3d

SingleIndexLike2d = SingleFlatIndexLike | SingleStackedIndexLike2d
ArrayIndexLike2d = ArrayFlatIndexLike[_S0Inv] | ArrayStackedIndexLike2d[_S0Inv]

SingleIndexLike3d = SingleFlatIndexLike | SingleStackedIndexLike3d
ArrayIndexLike3d = ArrayFlatIndexLike[_S0Inv] | ArrayStackedIndexLike3d[_S0Inv]
