from __future__ import annotations

from typing import Any, Generic, Literal, TypedDict, TypeGuard, TypeVar, overload

import numpy as np

from surface_potential_analysis.util.interpolation import (
    pad_ft_points,
)

_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)
_LCov = TypeVar("_LCov", bound=int, covariant=True)

BasisVector = np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]


class _LegacyPositionBasis(TypedDict, Generic[_LCov]):
    """Represents a basis in position space."""

    _type: Literal["position"]
    n: _LCov
    delta_x: BasisVector


class _LegacyMomentumBasis(TypedDict, Generic[_LCov]):
    """
    Represents a basis in momentum space.

    We use the convention outlined here https://austen.uk/courses/tqm/second-quantization/
    where
    Vk = sum_r V_r e^-ikr / sqrt(N)
    ie to convert a vector from postion to momentum basis you do the np.fft.fft
    with norm = ortho, and to convert a vector from momentum to position you do the
    inverse fft
    """

    _type: Literal["momentum"]
    n: _LCov
    delta_x: BasisVector


FundamentalBasis = _LegacyPositionBasis[_LCov] | _LegacyMomentumBasis[_LCov]

_PCov = TypeVar("_PCov", bound=FundamentalBasis[Any], covariant=True)
_PInv = TypeVar("_PInv", bound=FundamentalBasis[Any])


class _LegacyTruncatedBasis(TypedDict, Generic[_LCov, _PCov]):
    """Represents a basis with a reduced number of states."""

    _type: Literal["truncated"]
    n: _LCov
    parent: _PCov


class _LegacyExplicitBasis(TypedDict, Generic[_LCov, _PCov]):
    """Represents a basis with a reduced number of states, given explicitly."""

    _type: Literal["explicit"]
    vectors: np.ndarray[tuple[_LCov, int], np.dtype[np.complex_]]
    parent: _PCov


InheritedBasis = (
    _LegacyTruncatedBasis[_LCov, _PCov] | _LegacyExplicitBasis[_LCov, _PCov]
)


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
    basis: Basis[_L1Inv, Any], _type: Literal["position"]
) -> TypeGuard[_LegacyPositionBasis[_L1Inv]]:
    ...


@overload
def is_basis_type(
    basis: Basis[_L1Inv, Any], _type: Literal["momentum"]
) -> TypeGuard[_LegacyMomentumBasis[_L1Inv]]:
    ...


@overload
def is_basis_type(
    basis: Basis[_L1Inv, _PInv], _type: Literal["truncated"]
) -> TypeGuard[_LegacyTruncatedBasis[_L1Inv, _PInv]]:
    ...


@overload
def is_basis_type(
    basis: Basis[_L1Inv, _PInv], _type: Literal["explicit"]
) -> TypeGuard[_LegacyExplicitBasis[_L1Inv, _PInv]]:
    ...


def is_basis_type(
    basis: Basis[Any, Any],
    _type: Literal["explicit", "truncated", "position", "momentum"],
) -> bool:
    """Determine whether the type of the basis is _type."""
    return basis["_type"] == _type


def _get_fundamental_basis(
    basis: BasisWithLength[_L1Inv, Any]
) -> FundamentalBasis[_L1Inv]:
    if is_basis_type(basis, "explicit") or is_basis_type(basis, "truncated"):
        return basis["parent"]
    return basis  # type: ignore[return-value]


@overload
def as_fundamental_basis(
    basis: _LegacyMomentumBasis[_L1Inv],
) -> _LegacyMomentumBasis[_L1Inv]:
    ...


@overload
def as_fundamental_basis(
    basis: _LegacyTruncatedBasis[_L1Inv, _LegacyMomentumBasis[_L2Inv]]
) -> _LegacyMomentumBasis[_L1Inv]:
    ...


@overload
def as_fundamental_basis(
    basis: _LegacyTruncatedBasis[_L1Inv, _LegacyMomentumBasis[_L2Inv]]
    | _LegacyMomentumBasis[_L1Inv]
) -> _LegacyMomentumBasis[_L1Inv]:
    ...


def as_fundamental_basis(
    basis: _LegacyTruncatedBasis[_L1Inv, _LegacyMomentumBasis[_L2Inv]]
    | _LegacyMomentumBasis[_L1Inv]
) -> _LegacyMomentumBasis[_L1Inv]:
    """
    Given a truncated basis in momentum convert to a momentum basis of a lower resolution.

    Parameters
    ----------
    basis : TruncatedBasis[_L1Inv, MomentumBasis[_L2Inv]]

    Returns
    -------
    MomentumBasis[_L1Inv]
    """
    if is_basis_type(basis, "momentum"):
        return basis
    return {"_type": "momentum", "delta_x": basis["parent"]["delta_x"], "n": basis["n"]}  # type: ignore[typeddict-item]


def _as_explicit_position_basis(
    basis: Basis[Any, Any]
) -> _LegacyExplicitBasis[int, _LegacyPositionBasis[int]]:
    """
    Convert a basis into an explicit position basis.

    Parameters
    ----------
    basis : TruncatedBasis[_L1Inv, MomentumBasis[_L2Inv]] | MomentumBasis[_L1Inv] | TruncatedBasis[_L1Inv, PositionBasis[_L2Inv]] | PositionBasis[_L1Inv] | ExplicitBasis[_L1Inv, MomentumBasis[_L2Inv]] | ExplicitBasis[_L1Inv, PositionBasis[_L2Inv]]
        original basis

    Returns
    -------
    ExplicitBasis[_L1Inv, PositionBasis[_L2Inv]]
        explicit position basis
    """
    if is_basis_type(basis, "position"):
        return {
            "_type": "explicit",
            "parent": {
                "_type": "position",
                "delta_x": basis["delta_x"],
                "n": basis["n"],  # type: ignore[typeddict-item]
            },
            "vectors": np.eye(basis["n"], basis["n"]),
        }
    if is_basis_type(basis, "momentum"):
        return {
            "_type": "explicit",
            "parent": {
                "_type": "position",
                "delta_x": basis["delta_x"],
                "n": basis["n"],  # type: ignore[typeddict-item]
            },
            "vectors": np.fft.ifft(
                np.eye(basis["n"], basis["n"]), axis=1, norm="ortho"
            ),
        }
    if is_basis_type(basis, "truncated"):
        # TODO: position - what does this mean??
        return {
            "_type": "explicit",
            "parent": {
                "_type": "position",
                "delta_x": basis["parent"]["delta_x"],
                "n": basis["parent"]["n"],
            },  # pad_ft_points selects the relevant states in the truncated momentum basis
            "vectors": pad_ft_points(  # type: ignore[typeddict-item]
                np.fft.ifft(
                    np.eye(basis["parent"]["n"], basis["parent"]["n"]),
                    axis=1,
                    norm="ortho",
                ),
                s=(basis["n"],),
                axes=(0,),
            ),
        }
    if is_basis_type(basis, "explicit") and basis["parent"]["_type"] == "momentum":
        return {
            "_type": "explicit",
            "parent": {
                "_type": "position",
                "delta_x": basis["parent"]["delta_x"],
                "n": basis["parent"]["n"],
            },
            "vectors": np.fft.ifft(basis["vectors"], axis=1, norm="ortho"),
        }

    return basis  # type: ignore[return-value]


B = TypeVar("B", bound=Basis[Any, Any])


# ruff: noqa: D102
class LegacyBasisUtilAdapter(Generic[B]):
    """Helper class for dealing with Basis."""

    _basis: B

    def __init__(self, basis: B) -> None:
        self._basis = basis

    @property
    def n(self) -> int:
        if is_basis_type(self._basis, "explicit"):
            return self._basis["vectors"].shape[0]  # type: ignore[no-any-return]
        return self._basis["n"]  # type: ignore[no-any-return,typeddict-item]

    @property
    def fundamental_n(self) -> int:
        fundamental_basis = _get_fundamental_basis(self._basis)  # type: ignore[arg-type,var-annotated]
        return fundamental_basis["n"]  # type: ignore[no-any-return]

    @property
    def delta_x(self) -> BasisVector:
        fundamental_basis = _get_fundamental_basis(self._basis)  # type: ignore[var-annotated]
        return fundamental_basis["delta_x"]

    @property
    def vectors(self) -> np.ndarray[tuple[int, int], np.dtype[np.complex_]]:
        return _as_explicit_position_basis(self._basis)["vectors"]
