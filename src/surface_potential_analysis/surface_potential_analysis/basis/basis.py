from __future__ import annotations

from functools import cached_property
from typing import Any, Generic, Literal, TypedDict, TypeGuard, TypeVar, overload

import numpy as np

_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)
_LCov = TypeVar("_LCov", bound=int, covariant=True)

BasisVector = np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]


class PositionBasis(TypedDict, Generic[_LCov]):
    _type: Literal["position"]
    n: _LCov
    delta_x: BasisVector


class MomentumBasis(TypedDict, Generic[_LCov]):
    _type: Literal["momentum"]
    n: _LCov
    delta_x: BasisVector


FundamentalBasis = PositionBasis[_LCov] | MomentumBasis[_LCov]

_PCov = TypeVar("_PCov", bound=FundamentalBasis[Any], covariant=True)
_PInv = TypeVar("_PInv", bound=FundamentalBasis[Any])


class TruncatedBasis(TypedDict, Generic[_LCov, _PCov]):
    _type: Literal["truncated"]
    n: _LCov
    parent: _PCov


class ExplicitBasis(TypedDict, Generic[_LCov, _PCov]):
    _type: Literal["explicit"]
    vectors: np.ndarray[tuple[_LCov, int], np.dtype[np.complex_]]
    parent: _PCov


InheritedBasis = TruncatedBasis[_LCov, _PCov] | ExplicitBasis[_LCov, _PCov]


Basis = FundamentalBasis[_L1Inv] | InheritedBasis[_L1Inv, _PInv]
"""
The basis used to represent a Bloch wavefunction with _L1Inv states
"""

BasisWithLength = (
    FundamentalBasis[_L1Inv] | InheritedBasis[_L2Inv, FundamentalBasis[_L1Inv]]
)
"""
The basis used to represent a Bloch wavefunction with _L1Inv fundamental states
"""


@overload
def is_basis_type(
    basis: Basis[_L1Inv, Any], type: Literal["position"]
) -> TypeGuard[PositionBasis[_L1Inv]]:
    ...


@overload
def is_basis_type(
    basis: Basis[_L1Inv, Any], type: Literal["momentum"]
) -> TypeGuard[MomentumBasis[_L1Inv]]:
    ...


@overload
def is_basis_type(
    basis: Basis[_L1Inv, _PInv], type: Literal["truncated"]
) -> TypeGuard[TruncatedBasis[_L1Inv, _PInv]]:
    ...


@overload
def is_basis_type(
    basis: Basis[_L1Inv, _PInv], type: Literal["explicit"]
) -> TypeGuard[ExplicitBasis[_L1Inv, _PInv]]:
    ...


def is_basis_type(
    basis: Basis[Any, Any],
    type: Literal["explicit", "truncated", "position", "momentum"],
) -> bool:
    """Determines whether all objects in the list are strings"""
    return basis["_type"] == type


def get_fundamental_basis(
    basis: BasisWithLength[_L1Inv, Any]
) -> FundamentalBasis[_L1Inv]:
    if is_basis_type(basis, "explicit"):
        return basis["parent"]
    elif is_basis_type(basis, "truncated"):
        return basis["parent"]
    return basis  # type: ignore


B = TypeVar("B", bound=Basis[Any, Any])


class BasisUtil(Generic[B]):
    _basis: B

    def __init__(self, basis: B) -> None:
        self._basis = basis

    def __len__(self: BasisUtil[Basis[_L1Inv, Any]]) -> _L1Inv:
        if is_basis_type(self._basis, "explicit"):
            return self._basis["vectors"].shape[0]  # type:ignore
        return self._basis["n"]  # type:ignore

    @cached_property
    def fundamental_basis(self) -> FundamentalBasis[Any]:
        return get_fundamental_basis(self._basis)

    @overload
    def get_fundamental_basis_in(
        self, _type: Literal["position"]
    ) -> PositionBasis[Any]:
        ...

    @overload
    def get_fundamental_basis_in(
        self, _type: Literal["momentum"]
    ) -> MomentumBasis[Any]:
        ...

    def get_fundamental_basis_in(
        self, _type: Literal["position", "momentum"]
    ) -> FundamentalBasis[Any]:
        return {"_type": _type, "n": self.fundamental_n, "delta_x": self.delta_x}  # type: ignore

    @property
    def fundamental_n(self: BasisUtil[BasisWithLength[_L1Inv, Any]]) -> _L1Inv:
        return self.fundamental_basis["n"]  # type: ignore

    @property
    def fundamental_nk_points(
        self: BasisUtil[BasisWithLength[_L1Inv, Any]]
    ) -> np.ndarray[tuple[_L1Inv], np.dtype[np.int_]]:
        # We want points from (-self.Nk + 1) // 2 to (self.Nk - 1) // 2
        return np.fft.ifftshift(  # type:ignore
            np.arange((-self.fundamental_n + 1) // 2, (self.fundamental_n + 1) // 2)
        )

    @property
    def fundamental_nx_points(
        self: BasisUtil[BasisWithLength[_L1Inv, Any]]
    ) -> np.ndarray[tuple[_L1Inv], np.dtype[np.int_]]:
        return np.arange(0, self.fundamental_n, dtype=int)  # type:ignore

    @property
    def n(self: BasisUtil[BasisWithLength[_L1Inv, Any]]) -> _L1Inv:
        return self.__len__()  # type: ignore

    @property
    def nx_points(
        self: BasisUtil[BasisWithLength[_L1Inv, Any]]
    ) -> np.ndarray[tuple[_L1Inv], np.dtype[np.int_]]:
        return np.arange(0, self.n, dtype=int)  # type:ignore

    @property
    def nk_points(
        self: BasisUtil[BasisWithLength[_L1Inv, Any]]
    ) -> np.ndarray[tuple[_L1Inv], np.dtype[np.int_]]:
        return np.fft.ifftshift(  # type:ignore
            np.arange((-self.n + 1) // 2, (self.n + 1) // 2)
        )

    @property
    def delta_x(self) -> BasisVector:
        return self.fundamental_basis["delta_x"]

    @property
    def fundamental_dx(self) -> BasisVector:
        return self.delta_x / self.fundamental_n  # type: ignore

    @property
    def fundamental_x_points(
        self,
    ) -> np.ndarray[tuple[Literal[3], _LCov], np.dtype[np.float_]]:
        return self.fundamental_dx[:, np.newaxis] * self.fundamental_nx_points[np.newaxis, :]  # type: ignore


class PositionBasisUtil(BasisUtil[PositionBasis[Any]]):
    @property
    def dx(self) -> BasisVector:
        return self.delta_x / self.n  # type: ignore

    @property
    def x_points(self) -> np.ndarray[tuple[Literal[3], _LCov], np.dtype[np.float_]]:
        return self.dx[:, np.newaxis] * self.nx_points[np.newaxis, :]  # type: ignore
