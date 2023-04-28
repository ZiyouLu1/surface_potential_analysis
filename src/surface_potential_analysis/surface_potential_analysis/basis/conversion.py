from typing import Any, TypeVar, overload

import numpy as np

from surface_potential_analysis.basis.basis import (
    Basis,
    ExplicitBasis,
    MomentumBasis,
    PositionBasis,
    TruncatedBasis,
    is_basis_type,
)
from surface_potential_analysis.interpolation import pad_ft_points

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)

_BX0Inv = TypeVar("_BX0Inv", bound=Basis[Any, Any])
_BX1Inv = TypeVar("_BX1Inv", bound=Basis[Any, Any])


@overload
def as_explicit_position_basis(
    basis: MomentumBasis[_L0Inv],
) -> ExplicitBasis[_L0Inv, PositionBasis[_L0Inv]]:
    ...


@overload
def as_explicit_position_basis(
    basis: PositionBasis[_L0Inv],
) -> ExplicitBasis[_L0Inv, PositionBasis[_L0Inv]]:
    ...


@overload
def as_explicit_position_basis(
    basis: PositionBasis[_L0Inv] | MomentumBasis[_L0Inv],
) -> ExplicitBasis[_L0Inv, PositionBasis[_L0Inv]]:
    ...


@overload
def as_explicit_position_basis(
    basis: TruncatedBasis[_L0Inv, MomentumBasis[_L1Inv]]
) -> ExplicitBasis[_L0Inv, PositionBasis[_L1Inv]]:
    ...


@overload
def as_explicit_position_basis(
    basis: TruncatedBasis[_L0Inv, PositionBasis[_L1Inv]]
) -> ExplicitBasis[_L0Inv, PositionBasis[_L1Inv]]:
    ...


@overload
def as_explicit_position_basis(
    basis: TruncatedBasis[_L0Inv, PositionBasis[_L1Inv]]
    | TruncatedBasis[_L0Inv, MomentumBasis[_L1Inv]]
) -> ExplicitBasis[_L0Inv, PositionBasis[_L1Inv]]:
    ...


@overload
def as_explicit_position_basis(
    basis: ExplicitBasis[_L0Inv, MomentumBasis[_L1Inv]]
) -> ExplicitBasis[_L0Inv, PositionBasis[_L1Inv]]:
    ...


@overload
def as_explicit_position_basis(
    basis: ExplicitBasis[_L0Inv, PositionBasis[_L1Inv]]
) -> ExplicitBasis[_L0Inv, PositionBasis[_L1Inv]]:
    ...


@overload
def as_explicit_position_basis(
    basis: TruncatedBasis[_L0Inv, MomentumBasis[_L1Inv]]
    | MomentumBasis[_L0Inv]
    | TruncatedBasis[_L0Inv, PositionBasis[_L1Inv]]
    | PositionBasis[_L0Inv]
    | ExplicitBasis[_L0Inv, MomentumBasis[_L1Inv]]
    | ExplicitBasis[_L0Inv, PositionBasis[_L1Inv]]
) -> ExplicitBasis[_L0Inv, PositionBasis[_L1Inv]]:
    ...


def as_explicit_position_basis(
    basis: TruncatedBasis[_L0Inv, MomentumBasis[_L1Inv]]
    | MomentumBasis[_L0Inv]
    | TruncatedBasis[_L0Inv, PositionBasis[_L1Inv]]
    | PositionBasis[_L0Inv]
    | ExplicitBasis[_L0Inv, MomentumBasis[_L1Inv]]
    | ExplicitBasis[_L0Inv, PositionBasis[_L1Inv]]
) -> ExplicitBasis[_L0Inv, PositionBasis[_L1Inv]]:
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
                "n": basis["n"],  # type:ignore[typeddict-item]
            },
            "vectors": np.eye(basis["n"], basis["n"]),
        }
    if is_basis_type(basis, "momentum"):
        return {
            "_type": "explicit",
            "parent": {
                "_type": "position",
                "delta_x": basis["delta_x"],
                "n": basis["n"],  # type:ignore[typeddict-item]
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

    return basis  # type:ignore[return-value]


def get_basis_conversion_matrix(
    basis0: _BX0Inv, basis1: _BX1Inv
) -> np.ndarray[tuple[int, int], np.dtype[np.complex_]]:
    """
    Given two basis states, get the matrix to convert between the two states.

    Parameters
    ----------
    basis0 : _BC0Inv
        basis to convert from
    basis1 : _BC1Inv
        basis to convert to

    Returns
    -------
    np.ndarray[tuple[_L0Inv, _L0Inv], np.dtype[np.complex_]]
    """
    vectors0: np.ndarray[Any, Any] = as_explicit_position_basis(basis0)["vectors"]
    vectors1: np.ndarray[Any, Any] = as_explicit_position_basis(basis1)["vectors"]
    return np.dot(vectors0, np.conj(vectors1).T)  # type: ignore[no-any-return]
