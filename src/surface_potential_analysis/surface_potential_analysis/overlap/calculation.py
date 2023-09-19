from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np

from surface_potential_analysis.axis.axis_like import BasisLike, BasisWithLengthLike
from surface_potential_analysis.state_vector.conversion import (
    convert_state_vector_to_position_basis,
)
from surface_potential_analysis.util.decorators import timed
from surface_potential_analysis.wavepacket.eigenstate_conversion import (
    unfurl_wavepacket,
)

if TYPE_CHECKING:
    from surface_potential_analysis.axis.stacked_axis import (
        StackedBasisLike,
    )
    from surface_potential_analysis.overlap.overlap import Overlap3d, Overlap3dBasis
    from surface_potential_analysis.wavepacket.wavepacket import Wavepacket

    _B3d0Inv = TypeVar(
        "_B3d0Inv",
        bound=StackedBasisLike[
            tuple[
                BasisWithLengthLike[Any, Any, Literal[3]],
                BasisWithLengthLike[Any, Any, Literal[3]],
                BasisWithLengthLike[Any, Any, Literal[3]],
            ]
        ],
    )


_B0 = TypeVar("_B0", bound=BasisLike[Any, Any])
_BL0 = TypeVar("_BL0", bound=BasisWithLengthLike[Any, Any, Any])


@timed
def calculate_wavepacket_overlap(
    wavepacket_0: Wavepacket[
        StackedBasisLike[_B0, _B0, _B0], StackedBasisLike[_BL0, _BL0, _BL0]
    ],
    wavepacket_1: Wavepacket[
        StackedBasisLike[_B0, _B0, _B0], StackedBasisLike[_BL0, _BL0, _BL0]
    ],
) -> Overlap3d[Overlap3dBasis]:
    """
    Given two wavepackets in (the same) momentum basis calculate the overlap factor.

    Parameters
    ----------
    wavepacket_0 : Wavepacket[_NS0Inv, _NS1Inv, StackedAxisLike[tuple[MomentumBasis[_L0Inv], MomentumBasis[_L1Inv], _A3d0Inv]]
    wavepacket_1 : Wavepacket[_NS0Inv, _NS1Inv, StackedAxisLike[tuple[MomentumBasis[_L0Inv], MomentumBasis[_L1Inv], _A3d0Inv]]

    Returns
    -------
    Overlap[StackedAxisLike[tuple[PositionBasis[int], PositionBasis[int], _A3d0Inv]]
    """
    eigenstate_0 = convert_state_vector_to_position_basis(
        unfurl_wavepacket(wavepacket_0)
    )
    eigenstate_1 = convert_state_vector_to_position_basis(
        unfurl_wavepacket(wavepacket_1)
    )

    vector = np.conj(eigenstate_0["data"]) * (eigenstate_1["data"])
    return {"basis": eigenstate_0["basis"], "data": vector}  # type: ignore[typeddict-item]
