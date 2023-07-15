from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Self, TypeVar

import numpy as np

from surface_potential_analysis.axis.axis import FundamentalAxis
from surface_potential_analysis.axis.axis_like import AxisLike, AxisVector2d
from surface_potential_analysis.basis.util import AxisWithLengthBasisUtil
from surface_potential_analysis.wavepacket.localization import get_wavepacket_two_points

if TYPE_CHECKING:
    from surface_potential_analysis.wavepacket.wavepacket import Wavepacket

    pass


_L0Cov = TypeVar("_L0Cov", bound=int, covariant=True)

_AX0Inv = TypeVar("_AX0Inv", bound=AxisLike[Any, Any])
_AX1Inv = TypeVar("_AX1Inv", bound=AxisLike[Any, Any])


class TunnellingSimulationBandsAxis(FundamentalAxis[_L0Cov]):
    """
    Represents the bands axis of the simulation.

    States are localized in their respective unit cells, with
    an approximate central location given by locations as a fraction of
    the unit cell axis vectors.
    """

    def __init__(
        self,
        locations: np.ndarray[tuple[Literal[2], _L0Cov], np.dtype[np.float_]],
        unit_cell: tuple[AxisVector2d, AxisVector2d],
    ) -> None:
        self.locations = locations
        self.unit_cell = unit_cell
        super().__init__(self.locations.shape[1])

    @classmethod
    def from_wavepackets(
        cls,  # noqa: ANN102
        wavepacket_list: list[Wavepacket[Any, Any]],
    ) -> Self:
        """
        Generate a basis given a list of wavepackets.

        Returns
        -------
        Self
        """
        util = AxisWithLengthBasisUtil(wavepacket_list[0]["basis"])

        locations = np.zeros((2, len(wavepacket_list)))
        for i, w in enumerate(wavepacket_list):
            idx0, idx1 = get_wavepacket_two_points(w)
            location = np.average([idx0, idx1], axis=0)
            locations[:, i] = location[0:2]
        return cls(locations, tuple(util.delta_x[0:2, 0:2]))  # type: ignore[arg-type]


_AX2Inv = TypeVar("_AX2Inv", bound=TunnellingSimulationBandsAxis[Any])

TunnellingSimulationBasis = tuple[_AX0Inv, _AX1Inv, _AX2Inv]
"""
Basis used to represent the tunnelling simulation state

First two axes represent the shape of the supercell, last axis is the band
"""
