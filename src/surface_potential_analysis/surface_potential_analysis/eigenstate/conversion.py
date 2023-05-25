from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from surface_potential_analysis.basis_config.conversion import (
    basis_config_as_fundamental_momentum_basis_config,
    basis_config_as_fundamental_position_basis_config,
    convert_vector,
)
from surface_potential_analysis.basis_config.util import BasisConfigUtil
from surface_potential_analysis.util.interpolation import pad_ft_points

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis_like import BasisLike
    from surface_potential_analysis.basis_config.basis_config import (
        BasisConfig,
    )
    from surface_potential_analysis.eigenstate.eigenstate import (
        FundamentalMomentumBasisEigenstate,
        FundamentalPositionBasisEigenstate,
    )

    from .eigenstate import (
        Eigenstate,
    )

    _BC0Inv = TypeVar("_BC0Inv", bound=BasisConfig[Any, Any, Any])
    _BC1Inv = TypeVar("_BC1Inv", bound=BasisConfig[Any, Any, Any])

    _N0Inv = TypeVar("_N0Inv", bound=int)
    _N1Inv = TypeVar("_N1Inv", bound=int)
    _N2Inv = TypeVar("_N2Inv", bound=int)

    _NF0Inv = TypeVar("_NF0Inv", bound=int)
    _NF1Inv = TypeVar("_NF1Inv", bound=int)
    _NF2Inv = TypeVar("_NF2Inv", bound=int)


def convert_eigenstate_to_basis(
    eigenstate: Eigenstate[_BC0Inv], basis: _BC1Inv
) -> Eigenstate[_BC1Inv]:
    """
    Given an eigenstate, calculate the vector in the given basis.

    Parameters
    ----------
    eigenstate : Eigenstate[_BC0Inv]
    basis : _BC1Inv

    Returns
    -------
    Eigenstate[_BC1Inv]
    """
    converted = convert_vector(eigenstate["vector"], eigenstate["basis"], basis)
    return {"basis": basis, "vector": converted}  # type: ignore[typeddict-item]


def convert_eigenstate_to_position_basis(
    eigenstate: Eigenstate[
        BasisConfig[
            BasisLike[_NF0Inv, _N0Inv],
            BasisLike[_NF1Inv, _N1Inv],
            BasisLike[_NF2Inv, _N2Inv],
        ]
    ],
) -> FundamentalPositionBasisEigenstate[_NF0Inv, _NF1Inv, _NF2Inv]:
    """
    Given an eigenstate, calculate the vector in the given basis.

    Parameters
    ----------
    eigenstate : Eigenstate[_BC0Inv]
    basis : _BC1Inv

    Returns
    -------
    Eigenstate[_BC1Inv]
    """
    return convert_eigenstate_to_basis(
        eigenstate,
        basis_config_as_fundamental_position_basis_config(eigenstate["basis"]),
    )


def convert_eigenstate_to_momentum_basis(
    eigenstate: Eigenstate[
        BasisConfig[
            BasisLike[_NF0Inv, _N0Inv],
            BasisLike[_NF1Inv, _N1Inv],
            BasisLike[_NF2Inv, _N2Inv],
        ]
    ],
) -> FundamentalMomentumBasisEigenstate[_NF0Inv, _NF1Inv, _NF2Inv]:
    """
    Given an eigenstate, calculate the vector in the given basis.

    Parameters
    ----------
    eigenstate : Eigenstate[_BC0Inv]
    basis : _BC1Inv

    Returns
    -------
    Eigenstate[_BC1Inv]
    """
    return convert_eigenstate_to_basis(
        eigenstate,
        basis_config_as_fundamental_momentum_basis_config(eigenstate["basis"]),
    )


def convert_position_basis_eigenstate_to_momentum_basis(
    eigenstate: FundamentalPositionBasisEigenstate[_NF0Inv, _NF1Inv, _NF2Inv]
) -> FundamentalMomentumBasisEigenstate[_NF0Inv, _NF1Inv, _NF2Inv]:
    """
    convert an eigenstate from position to momentum basis.

    Parameters
    ----------
    eigenstate : PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]

    Returns
    -------
    MomentumBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]
    """
    util = BasisConfigUtil(eigenstate["basis"])
    transformed = np.fft.fftn(
        eigenstate["vector"].reshape(util.shape),
        axes=(0, 1, 2),
        s=util.fundamental_shape,
        norm="ortho",
    )
    return {
        "basis": basis_config_as_fundamental_momentum_basis_config(eigenstate["basis"]),
        "vector": transformed.reshape(-1),
    }


def convert_momentum_basis_eigenstate_to_position_basis(
    eigenstate: FundamentalMomentumBasisEigenstate[_NF0Inv, _NF1Inv, _NF2Inv]
) -> FundamentalPositionBasisEigenstate[_NF0Inv, _NF1Inv, _NF2Inv]:
    """
    Convert an eigenstate from momentum to position basis.

    Parameters
    ----------
    eigenstate : MomentumBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]

    Returns
    -------
    PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]
    """
    util = BasisConfigUtil(eigenstate["basis"])
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
        "basis": basis_config_as_fundamental_position_basis_config(eigenstate["basis"]),
        "vector": transformed.reshape(-1),
    }
