from __future__ import annotations

from typing import Any, TypeVarTuple

import numpy as np

# The 6 `<X>Like_co` type-aliases below represent all scalars that can be
# coerced into `<X>` (with the casting rule `same_kind`)
_BoolLike_co = bool | np.bool_
_UIntLike_co = _BoolLike_co | np.unsignedinteger[Any]
IntLike_co = int | np.integer[Any]
FloatLike_co = IntLike_co | float | np.floating[Any]
_ComplexLike_co = FloatLike_co | complex | np.complexfloating[Any, Any]
_TD64Like_co = IntLike_co | np.timedelta64

_TS = TypeVarTuple("_TS")

SingleFlatIndexLike = IntLike_co
ArrayFlatIndexLike = np.ndarray[tuple[*_TS], np.dtype[np.int_]]
FlatIndexLike = SingleFlatIndexLike | ArrayFlatIndexLike[Any]

SingleStackedIndexLike = tuple[IntLike_co, ...]
ArrayStackedIndexLike = tuple[np.ndarray[tuple[*_TS], np.dtype[np.int_]], ...]
StackedIndexLike = SingleStackedIndexLike | ArrayStackedIndexLike[Any]

SingleIndexLike = SingleFlatIndexLike | SingleStackedIndexLike
ArrayIndexLike = ArrayFlatIndexLike[tuple[*_TS]] | ArrayStackedIndexLike[tuple[*_TS]]
IndexLike = SingleIndexLike | ArrayIndexLike[Any]

SingleStackedIndexLike2d = tuple[IntLike_co, IntLike_co]
ArrayStackedIndexLike2d = tuple[
    np.ndarray[tuple[*_TS], np.dtype[np.int_]],
    np.ndarray[tuple[*_TS], np.dtype[np.int_]],
]
StackedIndexLike2d = SingleStackedIndexLike2d | ArrayStackedIndexLike2d[Any]
IndexLike2d = FlatIndexLike | StackedIndexLike2d

SingleStackedIndexLike3d = tuple[IntLike_co, IntLike_co, IntLike_co]
ArrayStackedIndexLike3d = tuple[
    np.ndarray[tuple[*_TS], np.dtype[np.int_]],
    np.ndarray[tuple[*_TS], np.dtype[np.int_]],
    np.ndarray[tuple[*_TS], np.dtype[np.int_]],
]
StackedIndexLike3d = SingleStackedIndexLike3d | ArrayStackedIndexLike3d[Any]
IndexLike3d = FlatIndexLike | StackedIndexLike3d

SingleIndexLike2d = SingleFlatIndexLike | SingleStackedIndexLike2d
ArrayIndexLike2d = (
    ArrayFlatIndexLike[tuple[*_TS]] | ArrayStackedIndexLike2d[tuple[*_TS]]
)

SingleIndexLike3d = SingleFlatIndexLike | SingleStackedIndexLike3d
ArrayIndexLike3d = (
    ArrayFlatIndexLike[tuple[*_TS]] | ArrayStackedIndexLike3d[tuple[*_TS]]
)


ShapeLike = tuple[IntLike_co, ...]
