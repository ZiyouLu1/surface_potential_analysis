from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from surface_potential_analysis.basis_config.conversion import (
    basis_config_as_fundamental_position_basis_config,
    convert_vector,
)
from surface_potential_analysis.util.decorators import timed

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis_like import BasisLike
    from surface_potential_analysis.basis_config.basis_config import (
        BasisConfig,
        FundamentalPositionBasisConfig,
    )

    from .wavepacket import Wavepacket

    _NS0Inv = TypeVar("_NS0Inv", bound=int)
    _NS1Inv = TypeVar("_NS1Inv", bound=int)

    _N0Inv = TypeVar("_N0Inv", bound=int)
    _N1Inv = TypeVar("_N1Inv", bound=int)
    _N2Inv = TypeVar("_N2Inv", bound=int)

    _NF0Inv = TypeVar("_NF0Inv", bound=int)
    _NF1Inv = TypeVar("_NF1Inv", bound=int)
    _NF2Inv = TypeVar("_NF2Inv", bound=int)

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
    wavepacket: Wavepacket[
        _NS0Inv,
        _NS1Inv,
        BasisConfig[
            BasisLike[_NF0Inv, _N0Inv],
            BasisLike[_NF1Inv, _N1Inv],
            BasisLike[_NF2Inv, _N2Inv],
        ],
    ]
) -> Wavepacket[
    _NS0Inv,
    _NS1Inv,
    FundamentalPositionBasisConfig[_NF0Inv, _NF1Inv, _NF2Inv],
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
    return convert_wavepacket_to_basis(
        wavepacket,
        basis_config_as_fundamental_position_basis_config(wavepacket["basis"]),
    )
