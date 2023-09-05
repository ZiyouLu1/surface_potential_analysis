from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Self, TypeVar

import numpy as np

from surface_potential_analysis.axis.axis import FundamentalAxis
from surface_potential_analysis.axis.axis_like import AxisLike, AxisVector2d
from surface_potential_analysis.basis.util import AxisWithLengthBasisUtil
from surface_potential_analysis.wavepacket.localization._tight_binding import (
    get_wavepacket_two_points,
)

if TYPE_CHECKING:
    from surface_potential_analysis.wavepacket.wavepacket import (
        WavepacketWithEigenvalues,
    )

    pass


_L0_co = TypeVar("_L0_co", bound=int, covariant=True)

_AX0Inv = TypeVar("_AX0Inv", bound=AxisLike[Any, Any])
_AX1Inv = TypeVar("_AX1Inv", bound=AxisLike[Any, Any])


class TunnellingSimulationBandsAxis(FundamentalAxis[_L0_co]):
    """
    Represents the bands axis of the simulation.

    States are localized in their respective unit cells, with
    an approximate central location given by locations as a fraction of
    the unit cell axis vectors.
    """

    def __init__(
        self,
        locations: np.ndarray[tuple[Literal[2], _L0_co], np.dtype[np.float_]],
        unit_cell: tuple[AxisVector2d, AxisVector2d],
    ) -> None:
        self.locations = locations
        self.unit_cell = unit_cell
        super().__init__(self.locations.shape[1])

    @classmethod
    def from_wavepackets(
        cls,  # noqa: ANN102
        wavepacket_list: list[WavepacketWithEigenvalues[Any, Any]],
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

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)

_L3Inv = TypeVar("_L3Inv", bound=int)


def get_basis_from_shape(
    shape: tuple[_L0Inv, _L1Inv],
    n_bands: _L2Inv,
    bands_axis: TunnellingSimulationBandsAxis[_L3Inv],
) -> tuple[
    FundamentalAxis[_L0Inv],
    FundamentalAxis[_L1Inv],
    TunnellingSimulationBandsAxis[_L2Inv],
]:
    """
    Get the simulation basis from the shape and TunnellingSimulationBandsAxis.

    Parameters
    ----------
    shape : tuple[_L0Inv, _L1Inv]
    bands_axis : TunnellingSimulationBandsAxis[_L2Inv]

    Returns
    -------
    tuple[FundamentalAxis[_L0Inv], FundamentalAxis[_L1Inv], TunnellingSimulationBandsAxis[_L2Inv]]
    """
    return (
        FundamentalAxis(shape[0]),
        FundamentalAxis(shape[1]),
        TunnellingSimulationBandsAxis(
            bands_axis.locations[:, :n_bands], bands_axis.unit_cell
        ),
    )
