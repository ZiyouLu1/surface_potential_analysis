from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from surface_potential_analysis.eigenstate.conversion import (
    convert_eigenstate_to_position_basis,
    convert_momentum_basis_eigenstate_to_position_basis,
)
from surface_potential_analysis.util import timed
from surface_potential_analysis.wavepacket.eigenstate_conversion import (
    unfurl_wavepacket,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import MomentumBasis, PositionBasis
    from surface_potential_analysis.basis_config.basis_config import (
        BasisConfig,
        MomentumBasisConfig,
        PositionBasisConfig,
    )
    from surface_potential_analysis.eigenstate.eigenstate import Eigenstate
    from surface_potential_analysis.overlap.overlap import Overlap
    from surface_potential_analysis.wavepacket import Wavepacket

    _BC0Inv = TypeVar("_BC0Inv", bound=BasisConfig[Any, Any, Any])

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)

_NS0Inv = TypeVar("_NS0Inv", bound=int)
_NS1Inv = TypeVar("_NS1Inv", bound=int)


def calculate_overlap_momentum_eigenstate(
    eigenstate_0: Eigenstate[MomentumBasisConfig[_L0Inv, _L1Inv, _L2Inv]],
    eigenstate_1: Eigenstate[MomentumBasisConfig[_L0Inv, _L1Inv, _L2Inv]],
) -> Overlap[PositionBasisConfig[_L0Inv, _L1Inv, _L2Inv]]:
    """
    Calculate the overlap between two eigenstates in position basis.

    Parameters
    ----------
    eigenstate_0 : Eigenstate[MomentumBasisConfig[_L0Inv,_L1Inv, _L2Inv]]
    eigenstate_1 : Eigenstate[MomentumBasisConfig[_L0Inv,_L1Inv, _L2Inv]]

    Returns
    -------
    Overlap[PositionBasisConfig[_L0Inv,_L1Inv, _L2Inv]]
    """
    converted_1 = convert_momentum_basis_eigenstate_to_position_basis(eigenstate_0)
    converted_2 = convert_momentum_basis_eigenstate_to_position_basis(eigenstate_1)

    vector = np.conj(converted_1["vector"]) * (converted_2["vector"])
    return {"basis": converted_1["basis"], "vector": vector}


@timed
def calculate_wavepacket_overlap(
    wavepacket_0: Wavepacket[
        _NS0Inv,
        _NS1Inv,
        BasisConfig[
            MomentumBasis[_L0Inv], MomentumBasis[_L1Inv], MomentumBasis[_L2Inv]
        ],
    ],
    wavepacket_1: Wavepacket[
        _NS0Inv,
        _NS1Inv,
        BasisConfig[
            MomentumBasis[_L0Inv], MomentumBasis[_L1Inv], MomentumBasis[_L2Inv]
        ],
    ],
) -> Overlap[
    BasisConfig[PositionBasis[int], PositionBasis[int], PositionBasis[_L2Inv]]
]:
    """
    Given two wavepackets in (the same) momentum basis calculate the overlap factor.

    Parameters
    ----------
    wavepacket_0 : Wavepacket[_NS0Inv, _NS1Inv, BasisConfig[MomentumBasis[_L0Inv], MomentumBasis[_L1Inv], _BX0Inv]]
    wavepacket_1 : Wavepacket[_NS0Inv, _NS1Inv, BasisConfig[MomentumBasis[_L0Inv], MomentumBasis[_L1Inv], _BX0Inv]]

    Returns
    -------
    Overlap[BasisConfig[PositionBasis[int], PositionBasis[int], _BX0Inv]]
    """
    eigenstate_0 = convert_momentum_basis_eigenstate_to_position_basis(
        unfurl_wavepacket(wavepacket_0)
    )
    eigenstate_1 = convert_momentum_basis_eigenstate_to_position_basis(
        unfurl_wavepacket(wavepacket_1)
    )

    vector = np.conj(eigenstate_0["vector"]) * (eigenstate_1["vector"])
    return {"basis": eigenstate_0["basis"], "vector": vector}


def calculate_overlap_eigenstate(
    eigenstate_0: Eigenstate[_BC0Inv],
    eigenstate_1: Eigenstate[_BC0Inv],
) -> Overlap[PositionBasisConfig[int, int, int]]:
    """
    Calculate the overlap between two eigenstates in position basis.

    Parameters
    ----------
    eigenstate_0 : Eigenstate[_BC0Inv]
    eigenstate_1 : Eigenstate[_BC0Inv]

    Returns
    -------
    Overlap[PositionBasisConfig[int, int, int]]
    """
    converted_0 = convert_eigenstate_to_position_basis(eigenstate_0)
    converted_1 = convert_eigenstate_to_position_basis(eigenstate_1)

    vector = np.conj(converted_0["vector"]) * (converted_1["vector"])
    return {"basis": converted_0["basis"], "vector": vector}
