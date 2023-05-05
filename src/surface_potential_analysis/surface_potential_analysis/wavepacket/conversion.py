from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from surface_potential_analysis.basis_config.basis_config import (
    BasisConfig,
    BasisConfigUtil,
)
from surface_potential_analysis.basis_config.conversion import convert_vector
from surface_potential_analysis.util import timed

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import (
        PositionBasis,
    )

    from .wavepacket import Wavepacket

    _NS0Inv = TypeVar("_NS0Inv", bound=int)
    _NS1Inv = TypeVar("_NS1Inv", bound=int)

    _BC0Inv = TypeVar("_BC0Inv", bound=BasisConfig[Any, Any, Any])
    _BC1Inv = TypeVar("_BC1Inv", bound=BasisConfig[Any, Any, Any])


@timed
def convert_wavepacket_to_basis(
    wavepacket: Wavepacket[_NS0Inv, _NS1Inv, _BC0Inv],
    basis: _BC1Inv,
) -> Wavepacket[_NS0Inv, _NS1Inv, _BC1Inv]:
    """
    Given a wavepacket convert it to the given basis.

    Parameters
    ----------
    wavepacket : Wavepacket[ _NS0Inv, _NS1Inv, _BC0Inv]
    basis : _BC1Inv

    Returns
    -------
    Wavepacket[_NS0Inv, _NS1Inv, _BC1Inv]
    """
    vectors = convert_vector(wavepacket["vectors"], wavepacket["basis"], basis)
    return {
        "basis": basis,
        "energies": wavepacket["energies"],
        "vectors": vectors,  # type:ignore[typeddict-item]
    }


def convert_wavepacket_to_position_basis(
    wavepacket: Wavepacket[_NS0Inv, _NS1Inv, _BC0Inv]
) -> Wavepacket[
    _NS0Inv,
    _NS1Inv,
    BasisConfig[PositionBasis[int], PositionBasis[int], PositionBasis[int]],
]:
    """
    Convert a wavepacket to the fundamental position basis.

    Parameters
    ----------
    wavepacket : Wavepacket[_NS0Inv, _NS1Inv, _BC0Inv]

    Returns
    -------
    Wavepacket[_NS0Inv, _NS1Inv, BasisConfig[PositionBasis[int], PositionBasis[int], PositionBasis[int]]]
    """
    util = BasisConfigUtil(wavepacket["basis"])
    return convert_wavepacket_to_basis(
        wavepacket, util.get_fundamental_basis_in("position")
    )
