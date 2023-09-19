from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Protocol,
    TypeVar,
    TypeVarTuple,
    Union,
    Unpack,
    overload,
    runtime_checkable,
)

import numpy as np

from surface_potential_analysis.axis.axis import (
    FundamentalBasis,
    FundamentalTransformedBasis,
)
from surface_potential_analysis.axis.axis_like import (
    BasisLike,
    convert_vector,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


_S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])

_B0 = TypeVarTuple("_B0")
_B1 = TypeVarTuple("_B1")
_B0Inv = TypeVar("_B0Inv", bound=BasisLike[Any, Any])


@runtime_checkable
class StackedBasisLike(BasisLike[Any, Any], Protocol[*_B0]):
    @property
    def ndim(self: StackedBasisLike[*tuple[_B0Inv, ...]]) -> int:
        return len(self.fundamental_shape)

    @property
    def fundamental_n(self: StackedBasisLike[*tuple[_B0Inv, ...]]) -> np.int_:
        return np.prod(self.fundamental_shape)

    @property
    def n(self: StackedBasisLike[*tuple[_B0Inv, ...]]) -> np.int_:
        return np.prod(self.shape)

    @property
    def shape(self: StackedBasisLike[*tuple[_B0Inv, ...]]) -> tuple[int, ...]:
        return tuple(ax.n for ax in self)

    @property
    def fundamental_shape(
        self: StackedBasisLike[*tuple[_B0Inv, ...]]
    ) -> tuple[int, ...]:
        return tuple(ax.fundamental_n for ax in self)

    def __repr__(self: StackedBasisLike[*tuple[_B0Inv, ...]]) -> str:
        return f"{self.__class__.__name__}({', '.join(b.__repr__() for b in self.__iter__())})"

    def __iter__(self) -> Iterator[Union[*_B0]]:
        ...

    @overload
    def __getitem__(
        self: StackedBasisLike[*tuple[Any, Any, _B0Inv, Unpack[tuple[Any, ...]]]],
        index: Literal[2],
    ) -> _B0Inv:
        ...

    @overload
    def __getitem__(
        self: StackedBasisLike[*tuple[Any, _B0Inv, Unpack[tuple[Any, ...]]]],
        index: Literal[1],
    ) -> _B0Inv:
        ...

    @overload
    def __getitem__(
        self: StackedBasisLike[*tuple[_B0Inv, Unpack[tuple[Any, ...]]]],
        index: Literal[0],
    ) -> _B0Inv:
        ...

    @overload
    def __getitem__(self: StackedBasisLike[*tuple[_B0Inv, ...]], index: int) -> _B0Inv:
        ...

    @overload  # TODO: return StackedAxisLike?
    def __getitem__(self, index: slice) -> StackedBasisLike[*tuple[Union[*_B0], ...]]:
        ...


def _convert_stacked_vector(
    vector: np.ndarray[_S0Inv, np.dtype[np.complex_] | np.dtype[np.float_]],
    initial_basis: StackedBasisLike[*tuple[BasisLike[Any, Any], ...]],
    final_basis: StackedBasisLike[*tuple[BasisLike[Any, Any], ...]],
    axis: int = -1,
) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
    """
    Convert a vector, expressed in terms of the given basis from_config in the basis to_config.

    Parameters
    ----------
    vector : np.ndarray[tuple[int], np.dtype[np.complex_] | np.dtype[np.float_]]
        the vector to convert
    from_config : _B3d0Inv
    to_config : _B3d1Inv
    axis : int, optional
        axis along which to convert, by default -1

    Returns
    -------
    np.ndarray[tuple[int], np.dtype[np.complex_]]
    """
    swapped = vector.swapaxes(axis, 0)
    stacked = swapped.reshape(*initial_basis.shape, *swapped.shape[1:])
    for ax, (initial, final) in enumerate(zip(initial_basis, final_basis, strict=True)):
        stacked = convert_vector(stacked, initial, final, ax)
    return (  # type: ignore[no-any-return]
        stacked.astype(np.complex_, copy=False)
        .reshape(-1, *swapped.shape[1:])
        .swapaxes(axis, 0)
    )


class StackedBasis(StackedBasisLike[Unpack[_B0]]):
    _axes: tuple[Unpack[_B0]]

    def __init__(self, *args: Unpack[_B0]) -> None:
        self._axes = args  # type: ignore[assignment]
        super().__init__()

    def __from_fundamental__(
        self: StackedBasisLike[*tuple[BasisLike[Any, Any], ...]],
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_] | np.dtype[np.float_]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
        basis = StackedBasis[Any](
            *tuple(FundamentalBasis(axis.fundamental_n) for axis in self)
        )
        return _convert_stacked_vector(vectors, basis, self, axis)

    def __into_fundamental__(
        self: StackedBasisLike[*tuple[BasisLike[Any, Any], ...]],
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_] | np.dtype[np.float_]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
        basis = StackedBasis[Any](
            *tuple(FundamentalBasis(axis.fundamental_n) for axis in self)
        )
        return _convert_stacked_vector(vectors, self, basis, axis)

    def __into_transformed__(
        self: StackedBasisLike[*tuple[BasisLike[Any, Any], ...]],
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_] | np.dtype[np.float_]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
        basis = StackedBasis[Any](
            *tuple(FundamentalTransformedBasis(axis.fundamental_n) for axis in self)
        )
        return _convert_stacked_vector(vectors, self, basis, axis)

    def __from_transformed__(
        self: StackedBasisLike[*tuple[BasisLike[Any, Any], ...]],
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_] | np.dtype[np.float_]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
        basis = StackedBasis[Any](
            *tuple(FundamentalTransformedBasis(axis.fundamental_n) for axis in self)
        )
        return _convert_stacked_vector(vectors, basis, self, axis)

    def __convert_vector_into__(
        self: StackedBasisLike[*tuple[BasisLike[Any, Any], ...]],
        vector: np.ndarray[_S0Inv, np.dtype[np.complex_] | np.dtype[np.float_]],
        basis: BasisLike[Any, Any],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[np.complex_]]:
        assert basis.fundamental_n == self.fundamental_n
        if not isinstance(basis, StackedBasisLike):
            return super().__convert_vector_into__(vector, basis, axis)
        # We overload __convert_vector_into__, more likely to get the 'happy path'
        return _convert_stacked_vector(vector, self, basis, axis)

    def __iter__(self) -> Iterator[Union[*_B0]]:
        return self._axes.__iter__()

    @overload
    def __getitem__(
        self: StackedBasisLike[*tuple[Any, Any, _B0Inv, Unpack[tuple[Any, ...]]]],
        index: Literal[2],
    ) -> _B0Inv:
        ...

    @overload
    def __getitem__(
        self: StackedBasisLike[*tuple[Any, _B0Inv, Unpack[tuple[Any, ...]]]],
        index: Literal[1],
    ) -> _B0Inv:
        ...

    @overload
    def __getitem__(
        self: StackedBasisLike[*tuple[_B0Inv, Unpack[tuple[Any, ...]]]],
        index: Literal[0],
    ) -> _B0Inv:
        ...

    @overload
    def __getitem__(self: StackedBasisLike[*tuple[_B0Inv, ...]], index: int) -> _B0Inv:
        ...

    @overload  # TODO: return StackedAxisLike?
    def __getitem__(self, index: slice) -> StackedBasisLike[*tuple[Union[*_B0], ...]]:
        ...

    def __getitem__(self, index: slice | int) -> Any:  # type: ignore typing ignores overload
        if isinstance(index, slice):
            return StackedBasis(*self._axes[index])
        return self._axes[index]
