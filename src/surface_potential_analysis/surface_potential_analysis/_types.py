from __future__ import annotations

from typing import Any, TypeVar, Union

import numpy as np

# The 6 `<X>Like_co` type-aliases below represent all scalars that can be
# coerced into `<X>` (with the casting rule `same_kind`)
_BoolLike_co = Union[bool, np.bool_]
_UIntLike_co = Union[_BoolLike_co, np.unsignedinteger]
_IntLike_co = Union[_BoolLike_co, int, np.integer]
_FloatLike_co = Union[_IntLike_co, float, np.floating]
_ComplexLike_co = Union[_FloatLike_co, complex, np.complexfloating]
_TD64Like_co = Union[_IntLike_co, np.timedelta64]

SingleFlatIndexLike = _IntLike_co
SingleStackedIndexLike = tuple[_IntLike_co, _IntLike_co, _IntLike_co]
SingleIndexLike = SingleFlatIndexLike | SingleStackedIndexLike

_S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])

ArrayFlatIndexLike = np.ndarray[_S0Inv, np.dtype[np.int_]]
ArrayStackedIndexLike = tuple[
    np.ndarray[_S0Inv, np.dtype[np.int_]],
    np.ndarray[_S0Inv, np.dtype[np.int_]],
    np.ndarray[_S0Inv, np.dtype[np.int_]],
]
ArrayIndexLike = ArrayFlatIndexLike[_S0Inv] | ArrayStackedIndexLike[_S0Inv]

IndexLike = SingleIndexLike | ArrayIndexLike[Any]
FlatIndexLike = SingleFlatIndexLike | ArrayFlatIndexLike[Any]
StackedIndexLike = SingleStackedIndexLike | ArrayStackedIndexLike[Any]
