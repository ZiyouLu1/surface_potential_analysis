from __future__ import annotations

from typing import Any, TypeVar

import numpy as np

from surface_potential_analysis.basis.basis import (
    ExplicitBasis,
    MomentumBasis,
    PositionBasis,
    TruncatedBasis,
    as_fundamental_basis,
)
from surface_potential_analysis.basis_config.basis_config import (
    BasisConfig,
)
from surface_potential_analysis.basis_config.conversion import convert_vector
from surface_potential_analysis.eigenstate.conversion import (
    convert_sho_eigenstate_to_momentum_basis,
)
from surface_potential_analysis.util import timed

from .wavepacket import Wavepacket, get_eigenstate

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)

_LF0Inv = TypeVar("_LF0Inv", bound=int)
_LF1Inv = TypeVar("_LF1Inv", bound=int)
_LF2Inv = TypeVar("_LF2Inv", bound=int)


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
    return {"basis": basis, "energies": wavepacket["energies"], "vectors": vectors}


@timed
def convert_sho_wavepacket_to_momentum(
    wavepacket: Wavepacket[
        _NS0Inv,
        _NS1Inv,
        BasisConfig[
            TruncatedBasis[_L0Inv, MomentumBasis[_LF0Inv]],
            TruncatedBasis[_L1Inv, MomentumBasis[_LF1Inv]],
            ExplicitBasis[_L2Inv, PositionBasis[_LF2Inv]],
        ],
    ]
) -> Wavepacket[
    _NS0Inv,
    _NS1Inv,
    BasisConfig[
        MomentumBasis[_L0Inv],
        MomentumBasis[_L1Inv],
        MomentumBasis[_LF2Inv],
    ],
]:
    """
    Convert a sho wavepacket to momentum basis.

    Parameters
    ----------
    wavepacket : Wavepacket[_NS0Inv, _NS1Inv, BasisConfig[ TruncatedBasis[_L0Inv, MomentumBasis[_LF0Inv]], TruncatedBasis[_L1Inv, MomentumBasis[_LF1Inv]], ExplicitBasis[_L2Inv, PositionBasis[_LF2Inv]]]]

    Returns
    -------
    Wavepacket[ _NS0Inv, _NS1Inv, BasisConfig[MomentumBasis[_LF0Inv], MomentumBasis[_LF1Inv], MomentumBasis[_LF2Inv]]]
    """
    vectors = np.array(
        [
            convert_sho_eigenstate_to_momentum_basis(get_eigenstate(wavepacket, i))[
                "vector"
            ]
            for i in range(wavepacket["energies"].size)
        ]
    ).reshape(*wavepacket["energies"].shape, -1)
    return {
        "basis": (
            as_fundamental_basis(wavepacket["basis"][0]),
            as_fundamental_basis(wavepacket["basis"][1]),
            {
                "_type": "momentum",
                "delta_x": wavepacket["basis"][2]["parent"]["delta_x"],
                "n": wavepacket["basis"][2]["parent"]["n"],
            },
        ),
        "vectors": vectors,
        "energies": wavepacket["energies"],
    }
