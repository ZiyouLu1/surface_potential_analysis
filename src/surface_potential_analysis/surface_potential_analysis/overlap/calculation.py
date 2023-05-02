from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from surface_potential_analysis.basis_config.basis_config import BasisConfigUtil
from surface_potential_analysis.eigenstate.conversion import (
    convert_eigenstate_to_basis,
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
    eigenstate_1: Eigenstate[MomentumBasisConfig[_L0Inv, _L1Inv, _L2Inv]],
    eigenstate2: Eigenstate[MomentumBasisConfig[_L0Inv, _L1Inv, _L2Inv]],
) -> Overlap[PositionBasisConfig[_L0Inv, _L1Inv, _L2Inv]]:
    """
    Calculate the overlap between two eigenstates in position basis.

    Parameters
    ----------
    eigenstate_1 : Eigenstate[MomentumBasisConfig[_L0Inv,_L1Inv, _L2Inv]]
    eigenstate2 : Eigenstate[MomentumBasisConfig[_L0Inv,_L1Inv, _L2Inv]]

    Returns
    -------
    Overlap[PositionBasisConfig[_L0Inv,_L1Inv, _L2Inv]]
    """
    converted1 = convert_momentum_basis_eigenstate_to_position_basis(eigenstate_1)
    converted2 = convert_momentum_basis_eigenstate_to_position_basis(eigenstate2)

    vector = np.conj(converted1["vector"]) * (converted2["vector"])
    return {"basis": converted1["basis"], "vector": vector}


@timed
def calculate_wavepacket_overlap(
    wavepacket1: Wavepacket[
        _NS0Inv,
        _NS1Inv,
        BasisConfig[
            MomentumBasis[_L0Inv], MomentumBasis[_L1Inv], MomentumBasis[_L2Inv]
        ],
    ],
    wavepacket2: Wavepacket[
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
    wavepacket1 : Wavepacket[_NS0Inv, _NS1Inv, BasisConfig[MomentumBasis[_L0Inv], MomentumBasis[_L1Inv], _BX0Inv]]
    wavepacket2 : Wavepacket[_NS0Inv, _NS1Inv, BasisConfig[MomentumBasis[_L0Inv], MomentumBasis[_L1Inv], _BX0Inv]]

    Returns
    -------
    Overlap[BasisConfig[PositionBasis[int], PositionBasis[int], _BX0Inv]]
    """
    eigenstate_1 = convert_momentum_basis_eigenstate_to_position_basis(
        unfurl_wavepacket(wavepacket1)
    )
    eigenstate2 = convert_momentum_basis_eigenstate_to_position_basis(
        unfurl_wavepacket(wavepacket2)
    )

    vector = np.conj(eigenstate_1["vector"]) * (eigenstate2["vector"])
    return {"basis": eigenstate_1["basis"], "vector": vector}


def calculate_overlap_eigenstate(
    eigenstate_1: Eigenstate[_BC0Inv],
    eigenstate2: Eigenstate[_BC0Inv],
) -> Overlap[PositionBasisConfig[int, int, int]]:
    """
    Calculate the overlap between two eigenstates in position basis.

    Parameters
    ----------
    eigenstate_1 : Eigenstate[MomentumBasisConfig[_L0Inv,_L1Inv, _L2Inv]]
    eigenstate2 : Eigenstate[MomentumBasisConfig[_L0Inv,_L1Inv, _L2Inv]]

    Returns
    -------
    Overlap[PositionBasisConfig[_L0Inv,_L1Inv, _L2Inv]]
    """
    util = BasisConfigUtil[Any, Any, Any](eigenstate_1["basis"])
    basis = util.get_fundamental_basis_in("position")
    converted1 = convert_eigenstate_to_basis(eigenstate_1, basis)
    converted2 = convert_eigenstate_to_basis(eigenstate2, basis)

    vector = np.conj(converted1["vector"]) * (converted2["vector"])
    return {"basis": basis, "vector": vector}
