from __future__ import annotations

from functools import cached_property
from typing import Any, Generic, Literal, TypeVar

import numpy as np

from surface_potential_analysis.basis.basis import (
    ExplicitBasis,
    FundamentalMomentumBasis,
    FundamentalPositionBasis,
)

from .basis_like import BasisLike, BasisVector

_BX0Inv = TypeVar("_BX0Inv", bound=BasisLike[Any, Any])


_N0Inv = TypeVar("_N0Inv", bound=int)
_N1Inv = TypeVar("_N1Inv", bound=int)

_NF0Inv = TypeVar("_NF0Inv", bound=int)
_NF1Inv = TypeVar("_NF1Inv", bound=int)


class _RotatedBasis(Generic[_NF0Inv, _N0Inv]):
    def __init__(
        self,
        basis: BasisLike[_NF0Inv, _N0Inv],
        matrix: np.ndarray[tuple[Literal[3], Literal[3]], np.dtype[np.float_]],
    ) -> None:
        self._basis = basis
        self._matrix = matrix

        self.__annotations__ = self._basis.__annotations__
        ##TODO: dunder methods

    def __getattr__(self, *args, **kwargs):  # type: ignore[no-untyped-def] # noqa: ANN204, ANN002, ANN003
        return getattr(self._basis, *args, **kwargs)

    @cached_property
    def delta_x(self) -> BasisVector:
        return np.dot(self._matrix, self._basis.delta_x)  # type: ignore[no-any-return]


def get_rotated_basis(
    basis: _BX0Inv,
    matrix: np.ndarray[tuple[Literal[3], Literal[3]], np.dtype[np.float_]],
) -> _BX0Inv:
    """
    Get the basis rotated by the given matrix.

    Parameters
    ----------
    basis : _BX0Inv
    matrix : np.ndarray[tuple[Literal[3], Literal[3]], np.dtype[np.float_]]

    Returns
    -------
    _BX0Inv
        The rotated basis
    """
    return _RotatedBasis(basis, matrix)  # type: ignore[return-value]


def get_basis_conversion_matrix(
    basis_0: BasisLike[_N0Inv, _NF0Inv], basis_1: BasisLike[_N1Inv, _NF1Inv]
) -> np.ndarray[tuple[_NF0Inv, _NF1Inv], np.dtype[np.complex_]]:
    """
    Get the matrix to convert one set of basis vectors into another.

    Parameters
    ----------
    basis_0 : BasisLike[_N0Inv, _NF0Inv]
    basis_1 : BasisLike[_N1Inv, _NF1Inv]

    Returns
    -------
    np.ndarray[tuple[_NF0Inv, _NF1Inv], np.dtype[np.complex_]]
    """
    vectors_0 = basis_0.vectors
    vectors_1 = basis_1.vectors
    return np.dot(vectors_0, np.conj(vectors_1).T)  # type: ignore[no-any-return]


def basis_as_fundamental_position_basis(
    basis: BasisLike[_NF0Inv, _N0Inv]
) -> FundamentalPositionBasis[_NF0Inv]:
    """
    Get the fundamental position basis for a given basis.

    Parameters
    ----------
    basis : BasisLike[_NF0Inv, _N0Inv]

    Returns
    -------
    FundamentalPositionBasis[_NF0Inv]
    """
    return FundamentalPositionBasis(basis.delta_x, basis.fundamental_n)


def basis_as_fundamental_momentum_basis(
    basis: BasisLike[_NF0Inv, _N0Inv]
) -> FundamentalMomentumBasis[_NF0Inv]:
    """
    Get the fundamental momentum basis for a given basis.

    Parameters
    ----------
    basis : BasisLike[_NF0Inv, _N0Inv]

    Returns
    -------
    FundamentalMomentumBasis[_NF0Inv]
    """
    return FundamentalMomentumBasis(basis.delta_x, basis.fundamental_n)


def basis_as_explicit_position_basis(
    basis: BasisLike[_NF0Inv, _N0Inv]
) -> ExplicitBasis[_NF0Inv, _N0Inv]:
    """
    Convert the basis into an explicit position basis.

    Parameters
    ----------
    basis : BasisLike[_NF0Inv, _N0Inv]

    Returns
    -------
    ExplicitBasis[_NF0Inv, _N0Inv]
    """
    return ExplicitBasis(basis.delta_x, basis.vectors)


def basis_as_single_point_basis(
    basis: BasisLike[_NF0Inv, _N0Inv]
) -> BasisLike[Literal[1], Literal[1]]:
    """
    Get the corresponding single point basis for a given basis.

    Parameters
    ----------
    basis : BasisLike[_NF0Inv, _N0Inv]

    Returns
    -------
    BasisLike[Literal[1], Literal[1]]
    """
    return FundamentalPositionBasis(basis.delta_x, 1)
