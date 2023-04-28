from __future__ import annotations

from typing import Any, TypeVar

import numpy as np

from surface_potential_analysis.basis.basis import Basis, PositionBasis
from surface_potential_analysis.basis_config.basis_config import (
    BasisConfig,
    BasisConfigUtil,
)
from surface_potential_analysis.util import timed
from surface_potential_analysis.wavepacket.conversion import convert_wavepacket_to_basis

from .wavepacket import Wavepacket, get_wavepacket_sample_fractions

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)

_LF0Inv = TypeVar("_LF0Inv", bound=int)
_LF1Inv = TypeVar("_LF1Inv", bound=int)
_LF2Inv = TypeVar("_LF2Inv", bound=int)

_BC0Inv = TypeVar("_BC0Inv", bound=BasisConfig[Any, Any, Any])

_BX0Inv = TypeVar("_BX0Inv", bound=Basis[Any, Any])
_BX1Inv = TypeVar("_BX1Inv", bound=Basis[Any, Any])
_BX2Inv = TypeVar("_BX2Inv", bound=Basis[Any, Any])

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


def _get_bloch_phase_position_basis(
    wavepacket: Wavepacket[
        _NS0Inv,
        _NS1Inv,
        BasisConfig[
            PositionBasis[_L0Inv], PositionBasis[_L1Inv], PositionBasis[_L2Inv]
        ],
    ],
    idx: int | tuple[int, int, int] = 0,
) -> np.ndarray[tuple[_NS0Inv, _NS1Inv], np.dtype[np.float_]]:
    """
    Get the phase of the bloch wavefunctions at the given point in position space.

    Parameters
    ----------
    wavepacket : Wavepacket[ _NS0Inv, _NS1Inv, BasisConfig[PositionBasis[_L0Inv], PositionBasis[_L1Inv], PositionBasis[_L2Inv]]]
        the wavepacket to calculate the phase of
    idx : int | tuple[int, int, int], optional
        the index in real space, by default 0

    Returns
    -------
    np.ndarray[tuple[_NS0Inv, _NS1Inv], np.dtype[np.float_]]
        the angle for each point in the wavepacket
    """
    eigenvectors = wavepacket["vectors"]
    # TODO use irreducible basis here?
    util = BasisConfigUtil(wavepacket["basis"])

    flat_idx = util.get_flat_index(idx) if isinstance(idx, tuple) else idx
    return np.angle(eigenvectors[:, :, flat_idx])  # type:ignore[return-value]


def normalize_wavepacket_position_basis(
    wavepacket: Wavepacket[
        _NS0Inv,
        _NS1Inv,
        BasisConfig[
            PositionBasis[_L0Inv], PositionBasis[_L1Inv], PositionBasis[_L2Inv]
        ],
    ],
    idx: int | tuple[int, int, int] = 0,
    angle: float = 0,
) -> Wavepacket[
    _NS0Inv,
    _NS1Inv,
    BasisConfig[PositionBasis[_L0Inv], PositionBasis[_L1Inv], PositionBasis[_L2Inv]],
]:
    """
    Normalize the eigenstates in a wavepacket.

    Parameters
    ----------
    wavepacket : Wavepacket[_NS0Inv, _NS1Inv, BasisConfig[PositionBasis[_L0Inv], PositionBasis[_L1Inv], PositionBasis[_L2Inv]]]
    idx : int | tuple[int, int, int], optional
        Index of the eigenstate to normalize, by default 0
        This index is taken in the irreducible unit cell
    angle: float, optional
        Angle to normalize the wavepacket to at the point idx.
        Each wavefunction will have the phase exp(i * angle) at the position
        given by idx

    Returns
    -------
    Wavepacket: Wavepacket[_NS0Inv, _NS1Inv, BasisConfig[PositionBasis[_L0Inv], PositionBasis[_L1Inv], PositionBasis[_L2Inv]]]
        The wavepacket, normalized
    """
    bloch_angles = _get_bloch_phase_position_basis(wavepacket, idx)
    global_phases = _get_global_phases(wavepacket, idx)

    phases = np.exp(-1j * (bloch_angles + global_phases - angle))
    fixed_eigenvectors = wavepacket["vectors"] * phases[:, :, np.newaxis]

    return {  # type: ignore[return-value]
        "basis": wavepacket["basis"],
        "vectors": fixed_eigenvectors,
        "energies": wavepacket["energies"],
    }


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
    flat_idx = util.get_flat_index(idx) if isinstance(idx, tuple) else idx

    return np.angle(converted["vectors"][:, :, flat_idx])  # type:ignore[return-value]


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
