from __future__ import annotations

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
    dk: BasisVector


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
            return self._basis["states"].shape[1]  # type:ignore
        return self._basis["n"]  # type:ignore

    @property
    def fundamental_basis(self) -> FundamentalBasis[Any]:
        return get_fundamental_basis(self._basis)

    @property
    def vector(self) -> BasisVector:
        if is_basis_type(self.fundamental_basis, "momentum"):
            return self.fundamental_basis["dk"]
        if is_basis_type(self.fundamental_basis, "position"):
            return self.fundamental_basis["delta_x"]
        raise AssertionError("Unreachable")

    @property
    def Nk(self: BasisUtil[BasisWithLength[_L1Inv, Any]]) -> _L1Inv:
        return self.fundamental_basis["n"]  # type: ignore

    @property
    def Nx(self: BasisUtil[BasisWithLength[_L1Inv, Any]]) -> _L1Inv:
        return self.fundamental_basis["n"]  # type: ignore

    @property
    def nk_points(
        self: BasisUtil[BasisWithLength[_L1Inv, Any]],
    ) -> np.ndarray[tuple[_L1Inv], np.dtype[np.int_]]:
        # We want points from (-self.Nk + 1) // 2 to (self.Nk - 1) // 2
        return np.fft.ifftshift(  # type:ignore
            np.arange((-self.Nk + 1) // 2, (self.Nk + 1) // 2)
        )

    @property
    def nx_points(
        self: BasisUtil[BasisWithLength[_L1Inv, Any]],
    ) -> np.ndarray[tuple[_L1Inv], np.dtype[np.int_]]:
        return np.arange(0, self.Nx, dtype=int)  # type:ignore


class MomentumBasisUtil(BasisUtil[MomentumBasis[_LCov]], Generic[_LCov]):
    @property
    def vector(self) -> BasisVector:
        return self.dk

    @property
    def dk(self) -> BasisVector:
        return self._basis["dk"]

    @property
    def k_points(self) -> np.ndarray[tuple[Literal[3], _LCov], np.dtype[np.float_]]:
        return self.dk * self.nk_points  # type: ignore


class PositionBasisUtil(BasisUtil[PositionBasis[_LCov]], Generic[_LCov]):
    @property
    def vector(self) -> BasisVector:
        return self.delta_x

    @property
    def delta_x(self) -> BasisVector:
        return self._basis["delta_x"]

    @property
    def dx(self) -> BasisVector:
        return self.delta_x / self.Nx  # type: ignore

    @property
    def x_points(self) -> np.ndarray[tuple[Literal[3], _LCov], np.dtype[np.float_]]:
        return self.dx * self.nx_points  # type: ignore
