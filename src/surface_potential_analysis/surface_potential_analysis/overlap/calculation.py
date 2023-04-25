from typing import Any, TypeVar

import numpy as np

from surface_potential_analysis.basis.basis import Basis, MomentumBasis, PositionBasis
from surface_potential_analysis.basis_config.basis_config import BasisConfig
from surface_potential_analysis.eigenstate.eigenstate_conversion import (
    convert_eigenstate_to_position_basis,
)
from surface_potential_analysis.overlap.overlap import Overlap
from surface_potential_analysis.wavepacket import Wavepacket
from surface_potential_analysis.wavepacket.eigenstate_conversion import (
    unfurl_wavepacket,
)

_BX0Inv = TypeVar("_BX0Inv", bound=Basis[Any, Any])

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)

_NS0Inv = TypeVar("_NS0Inv", bound=int)
_NS1Inv = TypeVar("_NS1Inv", bound=int)


def calculate_overlap(
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
    eigenstate1 = convert_eigenstate_to_position_basis(unfurl_wavepacket(wavepacket1))
    eigenstate2 = convert_eigenstate_to_position_basis(unfurl_wavepacket(wavepacket2))

    vector = np.conj(eigenstate1["vector"]) * (eigenstate2["vector"])
    return {"basis": eigenstate1["basis"], "vector": vector}
