from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from surface_potential_analysis.basis.conversion import (
    basis_as_fundamental_position_basis,
    convert_vector,
)
from surface_potential_analysis.util.decorators import timed

if TYPE_CHECKING:
    from surface_potential_analysis.axis.axis_like import AxisLike3d
    from surface_potential_analysis.basis.basis import (
        Basis,
        Basis3d,
        FundamentalPositionBasis3d,
    )
    from surface_potential_analysis.wavepacket.wavepacket import Wavepacket

    from .wavepacket import Wavepacket3dWith2dSamples

    _S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])

    _NS0Inv = TypeVar("_NS0Inv", bound=int)
    _NS1Inv = TypeVar("_NS1Inv", bound=int)

    _N0Inv = TypeVar("_N0Inv", bound=int)
    _N1Inv = TypeVar("_N1Inv", bound=int)
    _N2Inv = TypeVar("_N2Inv", bound=int)

    _NF0Inv = TypeVar("_NF0Inv", bound=int)
    _NF1Inv = TypeVar("_NF1Inv", bound=int)
    _NF2Inv = TypeVar("_NF2Inv", bound=int)

    _B0Inv = TypeVar("_B0Inv", bound=Basis[Any])
    _B1Inv = TypeVar("_B1Inv", bound=Basis[Any])


@timed
def convert_wavepacket_to_basis(
    wavepacket: Wavepacket[_S0Inv, _B0Inv],
    basis: _B1Inv,
) -> Wavepacket[_S0Inv, _B1Inv]:
    """
    Given a wavepacket convert it to the given basis.

    Parameters
    ----------
    wavepacket : Wavepacket[_S0Inv,  _B0Inv]
    basis : _B1Inv

    Returns
    -------
    Wavepacket[_S0Inv, _B1Inv]
    """
    vectors = convert_vector(wavepacket["vectors"], wavepacket["basis"], basis)
    return {
        "basis": basis,
        "energies": wavepacket["energies"],
        "shape": wavepacket["shape"],
        "vectors": vectors,  # type:ignore[typeddict-item]
    }


def convert_wavepacket_to_position_basis(
    wavepacket: Wavepacket3dWith2dSamples[
        _NS0Inv,
        _NS1Inv,
        Basis3d[
            AxisLike3d[_NF0Inv, _N0Inv],
            AxisLike3d[_NF1Inv, _N1Inv],
            AxisLike3d[_NF2Inv, _N2Inv],
        ],
    ]
) -> Wavepacket3dWith2dSamples[
    _NS0Inv,
    _NS1Inv,
    FundamentalPositionBasis3d[_NF0Inv, _NF1Inv, _NF2Inv],
]:
    """
    Convert a wavepacket to the fundamental position basis.

    Parameters
    ----------
    wavepacket : Wavepacket[_NS0Inv, _NS1Inv, _B3d0Inv]

    Returns
    -------
    Wavepacket[_NS0Inv, _NS1Inv, Basis3d[PositionBasis[int], PositionBasis[int], PositionBasis[int]]]
    """
    return convert_wavepacket_to_basis(
        wavepacket,
        basis_as_fundamental_position_basis(wavepacket["basis"]),
    )
