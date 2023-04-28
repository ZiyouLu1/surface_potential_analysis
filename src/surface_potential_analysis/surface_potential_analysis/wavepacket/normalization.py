from __future__ import annotations

from typing import Any, TypeVar

import numpy as np

from surface_potential_analysis.basis_config.basis_config import (
    BasisConfig,
    BasisConfigUtil,
)
from surface_potential_analysis.util import timed
from surface_potential_analysis.wavepacket.conversion import convert_wavepacket_to_basis

from .wavepacket import Wavepacket, get_wavepacket_sample_fractions

_BC0Inv = TypeVar("_BC0Inv", bound=BasisConfig[Any, Any, Any])

_NS0Inv = TypeVar("_NS0Inv", bound=int)
_NS1Inv = TypeVar("_NS1Inv", bound=int)


def _get_global_phases(
    wavepacket: Wavepacket[_NS0Inv, _NS1Inv, _BC0Inv],
    idx: int | tuple[int, int, int] = 0,
) -> np.ndarray[tuple[_NS0Inv, _NS1Inv], np.dtype[np.float_]]:
    """
    Get the global bloch phase at a given index in the irreducible cell.

    returns a list of the global phase of each eigenstate in the wavepacket

    Parameters
    ----------
    wavepacket : Wavepacket[_NS0Inv, _NS1Inv, BasisConfig[_BX0Inv, _BX1Inv, _BX2Inv]]
        The wavepacket to get the global phase for
    idx : int | tuple[int, int, int], optional
        The index in ravelled or unravelled form, by default 0

    Returns
    -------
    np.ndarray[tuple[_NS0Inv, _NS1Inv], np.dtype[np.float_]]
        list of list of phases for each is0, is1 sampled in the wavepacket
    """
    util = BasisConfigUtil[Any, Any, Any](wavepacket["basis"])
    stacked_idx = idx if isinstance(idx, tuple) else util.get_stacked_index(idx)
    # Total phase given by k.x = dk * j/Nj * i * delta_x / Ni
    #                          = 2 * pi * j/Nj * i / Ni
    # j / Nj
    momentum_fractions = get_wavepacket_sample_fractions(
        np.array(wavepacket["vectors"].shape[0:2])
    )
    # i / Ni
    position_fractions = np.array(
        [f / util.shape[i] for (i, f) in enumerate(stacked_idx[0:2])]
    )[:, np.newaxis, np.newaxis]

    return (  # type: ignore[no-any-return]
        2 * np.pi * np.sum(np.multiply(position_fractions, momentum_fractions), axis=0)
    )


@timed
def _get_bloch_phases(
    wavepacket: Wavepacket[_NS0Inv, _NS1Inv, _BC0Inv],
    idx: int | tuple[int, int, int] = 0,
) -> np.ndarray[tuple[_NS0Inv, _NS1Inv], np.dtype[np.float_]]:
    """
    Get the phase of the bloch wavefunctions at the given point in position space.

    Parameters
    ----------
    wavepacket : Wavepacket[ _NS0Inv, _NS1Inv, BasisConfig[PositionBasis[_L0Inv], PositionBasis[_L1Inv], _BX2Inv]]
        the wavepacket to calculate the phase of
    idx : int | tuple[int, int, int], optional
        the index in real space, by default 0

    Returns
    -------
    np.ndarray[tuple[_NS0Inv, _NS1Inv], np.dtype[np.float_]]
        the angle for each point in the wavepacket
    """
    basis = BasisConfigUtil(wavepacket["basis"]).get_fundamental_basis_in("position")
    converted = convert_wavepacket_to_basis(wavepacket, basis)
    util = BasisConfigUtil(converted["basis"])
    idx = util.get_flat_index(idx, mode="wrap") if isinstance(idx, tuple) else idx

    return np.angle(converted["vectors"][:, :, idx])  # type:ignore[return-value]


@timed
def normalize_wavepacket(
    wavepacket: Wavepacket[_NS0Inv, _NS1Inv, _BC0Inv],
    idx: int | tuple[int, int, int] = 0,
    angle: float = 0,
) -> Wavepacket[_NS0Inv, _NS1Inv, _BC0Inv]:
    """
    normalize a wavepacket in momentum basis.

    Parameters
    ----------
    wavepacket : Wavepacket[ _NS0Inv, _NS1Inv, BasisConfig[ TruncatedBasis[_L0Inv, MomentumBasis[_LF0Inv]], TruncatedBasis[_L1Inv, MomentumBasis[_LF1Inv]], ExplicitBasis[_L2Inv, PositionBasis[_LF2Inv]], ], ]
    idx : int | tuple[int, int, int], optional
        Index of the eigenstate to normalize, by default 0
        This index is taken in the irreducible unit cell
    angle: float, optional
        Angle to normalize the wavepacket to at the point idx.
        Each wavefunction will have the phase exp(i * angle) at the position
        given by idx

    Returns
    -------
    Wavepacket[ _NS0Inv, _NS1Inv, BasisConfig[ TruncatedBasis[_L0Inv, MomentumBasis[_LF0Inv]], TruncatedBasis[_L1Inv, MomentumBasis[_LF1Inv]], ExplicitBasis[_L2Inv, PositionBasis[_LF2Inv]], ], ]
    """
    bloch_angles = _get_bloch_phases(wavepacket, idx)
    global_phases = _get_global_phases(wavepacket, idx)

    phases = np.exp(-1j * (bloch_angles + global_phases - angle))
    fixed_eigenvectors = wavepacket["vectors"] * phases[:, :, np.newaxis]

    return {
        "basis": wavepacket["basis"],
        "vectors": fixed_eigenvectors,
        "energies": wavepacket["energies"],
    }


def calculate_normalisation(wavepacket: Wavepacket[_NS0Inv, _NS1Inv, _BC0Inv]) -> float:
    """
    calculate the normalization of a wavepacket.

    This should always be 1

    Parameters
    ----------
    wavepacket : Wavepacket[Any]

    Returns
    -------
    float
    """
    n_states = np.prod(wavepacket["energies"].shape)
    total_norm: complex = np.sum(np.conj(wavepacket["vectors"]) * wavepacket["vectors"])
    return float(total_norm / n_states)
