from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from surface_potential_analysis.basis.conversion import (
    basis_as_fundamental_position_basis,
    convert_vector,
)
from surface_potential_analysis.util.decorators import timed

if TYPE_CHECKING:
    from surface_potential_analysis.axis.axis import FundamentalPositionAxis
    from surface_potential_analysis.basis.basis import (
        Basis,
    )
    from surface_potential_analysis.wavepacket.wavepacket import Wavepacket

    _S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])

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
    wavepacket: Wavepacket[
        _S0Inv,
        _B0Inv,
    ]
) -> Wavepacket[_S0Inv, tuple[FundamentalPositionAxis[Any, Any], ...]]:
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
        wavepacket, basis_as_fundamental_position_basis(wavepacket["basis"])
    )
