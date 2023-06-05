from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from surface_potential_analysis.basis.conversion import (
    basis_as_fundamental_momentum_basis,
    basis_as_fundamental_position_basis,
    convert_vector,
)
from surface_potential_analysis.basis.util import Basis3dUtil
from surface_potential_analysis.util.decorators import timed
from surface_potential_analysis.util.interpolation import pad_ft_points

if TYPE_CHECKING:
    from surface_potential_analysis.axis.axis import FundamentalPositionAxis
    from surface_potential_analysis.axis.axis_like import AxisLike3d
    from surface_potential_analysis.basis.basis import (
        Basis,
        Basis3d,
    )
    from surface_potential_analysis.eigenstate.eigenstate import (
        Eigenstate,
        FundamentalMomentumBasisEigenstate3d,
        FundamentalPositionBasisEigenstate3d,
    )

    from .eigenstate import (
        Eigenstate3d,
    )

    _B0Inv = TypeVar("_B0Inv", bound=Basis[Any])
    _B1Inv = TypeVar("_B1Inv", bound=Basis[Any])

    _N0Inv = TypeVar("_N0Inv", bound=int)
    _N1Inv = TypeVar("_N1Inv", bound=int)
    _N2Inv = TypeVar("_N2Inv", bound=int)

    _NF0Inv = TypeVar("_NF0Inv", bound=int)
    _NF1Inv = TypeVar("_NF1Inv", bound=int)
    _NF2Inv = TypeVar("_NF2Inv", bound=int)


@timed
def convert_eigenstate_to_basis(
    eigenstate: Eigenstate[_B0Inv], basis: _B1Inv
) -> Eigenstate[_B1Inv]:
    """
    Given an eigenstate, calculate the vector in the given basis.

    Parameters
    ----------
    eigenstate : Eigenstate[_B0Inv]
    basis : _B1Inv

    Returns
    -------
    Eigenstate[_B1Inv]
    """
    converted = convert_vector(eigenstate["vector"], eigenstate["basis"], basis)
    return {"basis": basis, "vector": converted}  # type: ignore[typeddict-item]


def convert_eigenstate_to_position_basis(
    eigenstate: Eigenstate[_B0Inv],
) -> Eigenstate[tuple[FundamentalPositionAxis[Any, Any], ...]]:
    """
    Given an eigenstate, calculate the vector in position basis.

    Parameters
    ----------
    eigenstate : Eigenstate[_B0Inv]

    Returns
    -------
    Eigenstate[_B0Inv]
    """
    return convert_eigenstate_to_basis(
        eigenstate,
        basis_as_fundamental_position_basis(eigenstate["basis"]),
    )


def convert_eigenstate_to_momentum_basis(
    eigenstate: Eigenstate3d[
        Basis3d[
            AxisLike3d[_NF0Inv, _N0Inv],
            AxisLike3d[_NF1Inv, _N1Inv],
            AxisLike3d[_NF2Inv, _N2Inv],
        ]
    ],
) -> FundamentalMomentumBasisEigenstate3d[_NF0Inv, _NF1Inv, _NF2Inv]:
    """
    Given an eigenstate, calculate the vector in the given basis.

    Parameters
    ----------
    eigenstate : Eigenstate[_B3d0Inv]
    basis : _B3d1Inv

    Returns
    -------
    Eigenstate[_B3d1Inv]
    """
    return convert_eigenstate_to_basis(
        eigenstate,
        basis_as_fundamental_momentum_basis(eigenstate["basis"]),
    )


def convert_position_basis_eigenstate_to_momentum_basis(
    eigenstate: FundamentalPositionBasisEigenstate3d[_NF0Inv, _NF1Inv, _NF2Inv]
) -> FundamentalMomentumBasisEigenstate3d[_NF0Inv, _NF1Inv, _NF2Inv]:
    """
    convert an eigenstate from position to momentum basis.

    Parameters
    ----------
    eigenstate : PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]

    Returns
    -------
    MomentumBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]
    """
    util = Basis3dUtil(eigenstate["basis"])
    transformed = np.fft.fftn(
        eigenstate["vector"].reshape(util.shape),
        axes=(0, 1, 2),
        s=util.fundamental_shape,
        norm="ortho",
    )
    return {
        "basis": basis_as_fundamental_momentum_basis(eigenstate["basis"]),
        "vector": transformed.reshape(-1),
    }


def convert_momentum_basis_eigenstate_to_position_basis(
    eigenstate: FundamentalMomentumBasisEigenstate3d[_NF0Inv, _NF1Inv, _NF2Inv]
) -> FundamentalPositionBasisEigenstate3d[_NF0Inv, _NF1Inv, _NF2Inv]:
    """
    Convert an eigenstate from momentum to position basis.

    Parameters
    ----------
    eigenstate : MomentumBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]

    Returns
    -------
    PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]
    """
    util = Basis3dUtil(eigenstate["basis"])
    padded = pad_ft_points(
        eigenstate["vector"].reshape(util.shape),
        s=util.fundamental_shape,
        axes=(0, 1, 2),
    )
    transformed = np.fft.ifftn(
        padded,
        axes=(0, 1, 2),
        s=util.fundamental_shape,
        norm="ortho",
    )
    return {
        "basis": basis_as_fundamental_position_basis(eigenstate["basis"]),
        "vector": transformed.reshape(-1),
    }
