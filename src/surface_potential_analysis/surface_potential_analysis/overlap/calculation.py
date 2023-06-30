from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from surface_potential_analysis.state_vector.conversion import (
    convert_state_vector_to_position_basis,
)
from surface_potential_analysis.util.decorators import timed
from surface_potential_analysis.wavepacket.eigenstate_conversion import (
    unfurl_wavepacket,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import (
        Basis3d,
        FundamentalMomentumBasis3d,
        FundamentalPositionBasis3d,
    )
    from surface_potential_analysis.overlap.overlap import Overlap3d
    from surface_potential_analysis.state_vector.state_vector import StateVector3d
    from surface_potential_analysis.wavepacket.wavepacket import Wavepacket3d

    _B3d0Inv = TypeVar("_B3d0Inv", bound=Basis3d[Any, Any, Any])

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)

_NS0Inv = TypeVar("_NS0Inv", bound=int)
_NS1Inv = TypeVar("_NS1Inv", bound=int)
_S03dInv = TypeVar("_S03dInv", bound=tuple[int, int, int])


def calculate_overlap_momentum_eigenstate(
    eigenstate_0: StateVector3d[FundamentalMomentumBasis3d[_L0Inv, _L1Inv, _L2Inv]],
    eigenstate_1: StateVector3d[FundamentalMomentumBasis3d[_L0Inv, _L1Inv, _L2Inv]],
) -> Overlap3d[FundamentalPositionBasis3d[_L0Inv, _L1Inv, _L2Inv]]:
    """
    Calculate the overlap between two eigenstates in position basis.

    Parameters
    ----------
    eigenstate_0 : Eigenstate[MomentumBasis3d[_L0Inv,_L1Inv, _L2Inv]]
    eigenstate_1 : Eigenstate[MomentumBasis3d[_L0Inv,_L1Inv, _L2Inv]]

    Returns
    -------
    Overlap[PositionBasis3d[_L0Inv,_L1Inv, _L2Inv]]
    """
    converted_1 = convert_state_vector_to_position_basis(eigenstate_0)
    converted_2 = convert_state_vector_to_position_basis(eigenstate_1)

    vector = np.conj(converted_1["vector"]) * (converted_2["vector"])
    return {"basis": converted_1["basis"], "vector": vector}  # type: ignore[typeddict-item]


@timed
def calculate_wavepacket_overlap(
    wavepacket_0: Wavepacket3d[_S03dInv, _B3d0Inv],
    wavepacket_1: Wavepacket3d[_S03dInv, _B3d0Inv],
) -> Overlap3d[FundamentalPositionBasis3d[int, int, int]]:
    """
    Given two wavepackets in (the same) momentum basis calculate the overlap factor.

    Parameters
    ----------
    wavepacket_0 : Wavepacket[_NS0Inv, _NS1Inv, Basis3d[MomentumBasis[_L0Inv], MomentumBasis[_L1Inv], _A3d0Inv]]
    wavepacket_1 : Wavepacket[_NS0Inv, _NS1Inv, Basis3d[MomentumBasis[_L0Inv], MomentumBasis[_L1Inv], _A3d0Inv]]

    Returns
    -------
    Overlap[Basis3d[PositionBasis[int], PositionBasis[int], _A3d0Inv]]
    """
    eigenstate_0 = convert_state_vector_to_position_basis(
        unfurl_wavepacket(wavepacket_0)
    )
    eigenstate_1 = convert_state_vector_to_position_basis(
        unfurl_wavepacket(wavepacket_1)
    )

    vector = np.conj(eigenstate_0["vector"]) * (eigenstate_1["vector"])
    return {"basis": eigenstate_0["basis"], "vector": vector}  # type: ignore[typeddict-item]


def calculate_overlap_eigenstate(
    eigenstate_0: StateVector3d[_B3d0Inv],
    eigenstate_1: StateVector3d[_B3d0Inv],
) -> Overlap3d[FundamentalPositionBasis3d[int, int, int]]:
    """
    Calculate the overlap between two eigenstates in position basis.

    Parameters
    ----------
    eigenstate_0 : Eigenstate[_B3d0Inv]
    eigenstate_1 : Eigenstate[_B3d0Inv]

    Returns
    -------
    Overlap[PositionBasis3d[int, int, int]]
    """
    converted_0 = convert_state_vector_to_position_basis(eigenstate_0)  # type: ignore[arg-type,var-annotated]
    converted_1 = convert_state_vector_to_position_basis(eigenstate_1)  # type: ignore[arg-type,var-annotated]

    vector = np.conj(converted_0["vector"]) * (converted_1["vector"])
    return {"basis": converted_0["basis"], "vector": vector}  # type: ignore[typeddict-item]
