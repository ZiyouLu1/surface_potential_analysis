from __future__ import annotations

from itertools import starmap
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np

from surface_potential_analysis.basis.basis import (
    FundamentalPositionBasis,
)
from surface_potential_analysis.basis.basis_like import (
    AxisVector,
    BasisLike,
    BasisWithLengthLike,
)
from surface_potential_analysis.basis.conversion import basis_as_fundamental_basis
from surface_potential_analysis.basis.stacked_basis import StackedBasis
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.state_vector.conversion import (
    convert_state_vector_list_to_basis,
)
from surface_potential_analysis.state_vector.state_vector_list import (
    StateVectorList,
    get_basis_states,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.stacked_basis import StackedBasisLike

_NF0_co = TypeVar("_NF0_co", bound=int, covariant=True)
_N0_co = TypeVar("_N0_co", bound=int, covariant=True)

_NF0Inv = TypeVar("_NF0Inv", bound=int)
_N0Inv = TypeVar("_N0Inv", bound=int)
_ND0Inv = TypeVar("_ND0Inv", bound=int)

_S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])


class ExplicitBasis(BasisLike[_NF0_co, _N0_co]):
    """An basis with vectors given as explicit states."""

    def __init__(
        self,
        vectors: np.ndarray[tuple[_N0_co, _NF0_co], np.dtype[np.complex128]],
    ) -> None:
        self._vectors = vectors
        super().__init__()

    @property
    def n(self) -> _N0_co:
        return self.vectors.shape[0]  # type: ignore[no-any-return]

    @property
    def fundamental_n(self) -> _NF0_co:
        return self.vectors.shape[1]  # type: ignore[no-any-return]

    @property
    def vectors(self) -> np.ndarray[tuple[_N0_co, _NF0_co], np.dtype[np.complex128]]:
        return self._vectors

    @classmethod
    def from_state_vectors(
        cls: type[ExplicitBasis[_NF0_co, _N0_co]],
        vectors: StateVectorList[Any, Any],
    ) -> ExplicitBasis[_NF0_co, _N0_co]:
        converted = convert_state_vector_list_to_basis(
            vectors, basis_as_fundamental_basis(vectors["basis"][1])
        )["data"].reshape(vectors["basis"][0].n, -1)
        return cls(converted)

    @classmethod
    def from_basis(
        cls: type[ExplicitBasis[_NF0_co, _N0_co]],
        basis: BasisLike[_NF0_co, _N0_co],
    ) -> ExplicitBasis[_NF0_co, _N0_co]:
        return cls.from_state_vectors(get_basis_states(basis))

    @property
    def _transformed_vectors(
        self,
    ) -> np.ndarray[tuple[_N0_co, _NF0_co], np.dtype[np.complex128]]:
        return np.fft.fft(self.vectors, self.fundamental_n, axis=1, norm="ortho")

    def __from_fundamental__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex128] | np.dtype[np.float64]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
        transformed = np.tensordot(np.conj(self.vectors), vectors, axes=([0], [axis]))
        return np.moveaxis(transformed, 0, axis)

    def __into_fundamental__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex128] | np.dtype[np.float64]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
        transformed = np.tensordot(vectors, self.vectors, axes=([axis], [1]))
        return np.moveaxis(transformed, -1, axis)

    def __from_transformed__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex128] | np.dtype[np.float64]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
        transformed_vectors = np.conj(self._transformed_vectors)
        transformed = np.tensordot(transformed_vectors, vectors, axes=([0], [axis]))
        return np.moveaxis(transformed, 0, axis)

    def __into_transformed__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex128] | np.dtype[np.float64]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
        transformed_vectors = self._transformed_vectors
        transformed = np.tensordot(vectors, transformed_vectors, axes=([axis], [1]))
        return np.moveaxis(transformed, -1, axis)

    def __convert_vector_into__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex128] | np.dtype[np.float64]],
        basis: BasisLike[Any, Any],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[np.complex128]]:
        if isinstance(basis, ExplicitBasis):
            assert basis.fundamental_n == self.fundamental_n
            # We dont need to go all the way to fundamental basis here
            # Instead we can just compute the transformation once
            matrix = np.tensordot(np.conj(basis.vectors), self.vectors, axes=([1], [1]))
            transformed = np.tensordot(matrix, vectors, axes=([1], [axis]))
            return np.moveaxis(transformed, 0, axis)
        return super().__convert_vector_into__(vectors, basis, axis)


class ExplicitBasisWithLength(
    ExplicitBasis[_NF0_co, _N0_co], BasisWithLengthLike[_NF0_co, _N0_co, _ND0Inv]
):
    """An basis with vectors given as explicit states."""

    def __init__(
        self,
        delta_x: AxisVector[_ND0Inv],
        vectors: np.ndarray[tuple[_N0_co, _NF0_co], np.dtype[np.complex128]],
    ) -> None:
        self._delta_x = delta_x
        self._vectors = vectors
        super().__init__(vectors)

    @property
    def delta_x(self) -> AxisVector[_ND0Inv]:
        return self._delta_x


def basis_as_explicit_position_basis(
    axis: BasisWithLengthLike[_NF0Inv, _N0Inv, _ND0Inv],
) -> ExplicitBasisWithLength[_NF0Inv, _N0Inv, _ND0Inv]:
    """
    Convert the axis into an explicit position axis.

    Parameters
    ----------
    axis : BasisLike[_NF0Inv, _N0Inv]

    Returns
    -------
    ExplicitBasis[_NF0Inv, _N0Inv]
    """
    util = BasisUtil(axis)
    return ExplicitBasisWithLength(axis.delta_x, util.vectors)


def basis_as_orthonormal_basis(
    axis: BasisWithLengthLike[_NF0Inv, _N0Inv, _ND0Inv],
) -> ExplicitBasisWithLength[_NF0Inv, _N0Inv, _ND0Inv]:
    """
    make the given axis orthonormal.

    Parameters
    ----------
    axis : BasisLike[_NF0Inv, _N0Inv]

    Returns
    -------
    ExplicitBasis[_NF0Inv, _N0Inv]
    """
    vectors = BasisUtil(axis).vectors
    orthonormal_vectors = np.zeros_like(vectors, dtype=vectors.dtype)
    for i, v in enumerate(vectors):
        vector = v
        for other in orthonormal_vectors[:i]:
            vector -= np.dot(vector, other) * other
        orthonormal_vectors[i] = vector / np.linalg.norm(vector)

    return ExplicitBasisWithLength(axis.delta_x, orthonormal_vectors)


class ExplicitStackedBasisWithLength(
    ExplicitBasis[_NF0_co, _N0_co], Generic[_NF0_co, _N0_co, _ND0Inv]
):
    """An basis with vectors given as explicit states."""

    def __init__(
        self,
        delta_x_stacked: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        fundamental_shape: tuple[int, ...],
        vectors: np.ndarray[tuple[_N0_co, _NF0_co], np.dtype[np.complex128]],
    ) -> None:
        self._fundamental_shape = fundamental_shape
        self._delta_x = delta_x_stacked
        super().__init__(vectors)

    @property
    def delta_x_stacked(self) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        return self.delta_x_stacked

    @property
    def fundamental_shape(self) -> tuple[int, ...]:
        return self._fundamental_shape

    @classmethod
    def from_state_vectors_with_shape(
        cls: type[ExplicitStackedBasisWithLength[_NF0_co, _N0_co, _ND0Inv]],
        vectors: StateVectorList[Any, Any],
        *,
        delta_x_stacked: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        fundamental_shape: tuple[int, ...],
    ) -> ExplicitStackedBasisWithLength[_NF0_co, _N0_co, _ND0Inv]:
        converted = convert_state_vector_list_to_basis(
            vectors, basis_as_fundamental_basis(vectors["basis"][1])
        )["data"].reshape(vectors["basis"][0].n, -1)
        return cls(delta_x_stacked, fundamental_shape, converted)


def explicit_stacked_basis_as_fundamental(
    basis: ExplicitStackedBasisWithLength[Any, Any, Any],
) -> StackedBasisLike[*tuple[FundamentalPositionBasis[Any, Any], ...]]:
    """
    Get the fundamental basis for a given explicit stacked basis.

    Returns
    -------
    StackedBasisLike[*tuple[FundamentalPositionBasis[Any, Any], ...]]
    """
    return StackedBasis(
        *tuple(
            starmap(
                FundamentalPositionBasis[Any, Any],
                zip(basis.delta_x_stacked, basis.fundamental_shape),
            )
        )
    )
