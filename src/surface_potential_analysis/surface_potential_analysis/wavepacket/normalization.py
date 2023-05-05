from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, Unpack, overload

import numpy as np

from surface_potential_analysis.basis_config.basis_config import (
    BasisConfig,
    BasisConfigUtil,
    get_x01_mirrored_index,
    wrap_index_around_origin_x01,
)
from surface_potential_analysis.eigenstate.conversion import (
    convert_eigenstate_to_position_basis,
)
from surface_potential_analysis.util import timed
from surface_potential_analysis.wavepacket.conversion import (
    convert_wavepacket_to_position_basis,
)

from .wavepacket import Wavepacket, get_eigenstate, get_wavepacket_sample_fractions

if TYPE_CHECKING:
    from surface_potential_analysis._types import (
        ArrayIndexLike,
        IndexLike,
        SingleIndexLike,
    )

_BC0Inv = TypeVar("_BC0Inv", bound=BasisConfig[Any, Any, Any])

_NS0Inv = TypeVar("_NS0Inv", bound=int)
_NS1Inv = TypeVar("_NS1Inv", bound=int)

_S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])


@overload
def _get_global_phases(
    wavepacket: Wavepacket[_NS0Inv, _NS1Inv, _BC0Inv],
    idx: SingleIndexLike = 0,
) -> np.ndarray[tuple[_NS0Inv, _NS1Inv], np.dtype[np.float_]]:
    ...


@overload
def _get_global_phases(
    wavepacket: Wavepacket[_NS0Inv, _NS1Inv, _BC0Inv],
    idx: ArrayIndexLike[_S0Inv],
) -> np.ndarray[tuple[_NS0Inv, _NS1Inv, Unpack[_S0Inv]], np.dtype[np.float_]]:
    ...


def _get_global_phases(
    wavepacket: Wavepacket[_NS0Inv, _NS1Inv, _BC0Inv],
    idx: IndexLike = 0,
) -> (
    np.ndarray[tuple[_NS0Inv, _NS1Inv, Unpack[tuple[int, ...]]], np.dtype[np.float_]]
    | np.ndarray[tuple[_NS0Inv, _NS1Inv], np.dtype[np.float_]]
):
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
    basis = BasisConfigUtil(wavepacket["basis"]).get_fundamental_basis_in("position")
    util = BasisConfigUtil(basis)
    idx = idx if isinstance(idx, tuple) else util.get_stacked_index(idx)
    # Total phase given by k.x = dk * j/Nj * i * delta_x / Ni
    #                          = 2 * pi * j/Nj * i / Ni
    # j / Nj
    k_fractions = get_wavepacket_sample_fractions(
        np.array(wavepacket["vectors"].shape[0:2])
    )
    # i / Ni
    x0_fractions = np.asarray(idx[0] / util.n0)
    x1_fractions = np.asarray(idx[1] / util.n1)

    # Slices required to cast to [_NS0Inv, _NS1Inv, ..shape]
    k_slice = (
        slice(None),
        slice(None),
        *[np.newaxis for _ in np.array(idx).shape[1:]],
    )
    x_slice = (np.newaxis, np.newaxis)

    x0_phase = np.multiply(x0_fractions[*x_slice], k_fractions[0, *k_slice])
    x1_phase = np.multiply(x1_fractions[*x_slice], k_fractions[1, *k_slice])

    return 2 * np.pi * (x0_phase + x1_phase)  # type: ignore[no-any-return]


@timed
def _get_bloch_phases(
    wavepacket: Wavepacket[_NS0Inv, _NS1Inv, _BC0Inv],
    idx: SingleIndexLike = 0,
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
    converted = convert_wavepacket_to_position_basis(wavepacket)
    util = BasisConfigUtil(converted["basis"])
    idx = util.get_flat_index(idx, mode="wrap") if isinstance(idx, tuple) else idx

    return np.angle(converted["vectors"][:, :, idx])  # type: ignore[return-value]


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


def _get_bloch_phases_two_point(
    wavepacket: Wavepacket[_NS0Inv, _NS1Inv, _BC0Inv]
) -> np.ndarray[tuple[_NS0Inv, _NS1Inv], np.dtype[np.float_]]:
    converted = convert_wavepacket_to_position_basis(wavepacket)

    max_idx: np.ndarray[tuple[_NS0Inv, _NS1Inv], np.dtype[np.int_]] = np.argmax(
        np.abs(converted["vectors"]), axis=-1
    )
    mirror_idx = get_x01_mirrored_index(converted["basis"], max_idx)

    max_point = converted["vectors"][max_idx]
    mirror_point = converted["vectors"][mirror_idx]
    return 0.5 * (np.angle(max_point) + np.angle(mirror_point))  # type: ignore[no-any-return]


def _wrap_phases(
    phases: np.ndarray[tuple[int], np.dtype[np.float_]], half_width: float = np.pi
) -> np.ndarray[tuple[int], np.dtype[np.float_]]:
    return (phases + half_width) % (2 * half_width) - half_width  # type: ignore[no-any-return]


@timed
def normalize_wavepacket_two_point(
    wavepacket: Wavepacket[_NS0Inv, _NS1Inv, _BC0Inv],
    angle: float = 0,
) -> Wavepacket[_NS0Inv, _NS1Inv, _BC0Inv]:
    """
    Normalize a wavepacket using a 'two-point' calculation.

    This makes use of the x0=x1 symmetry in the wavepacket to calculate the angle of a wavepacket at the origin.
    We find that although the wavepacket may be zero at the origin, the angle changes sharply at the transition through
    the origin. Unfortunately this process doesn't work - to get the best localisation
    for a wavepacket which has a node in the origin we need to have one lobe with a phase of 0 and
    the other with a phase of pi (this produces +-pi/2 at both)
    For now we only consider the max location of the k=0 bloch wavefunction.

    Parameters
    ----------
    wavepacket : Wavepacket[_NS0Inv, _NS1Inv, _BC0Inv]
        Initial wavepacket
    angle : float, optional
        angle at the origin, by default 0

    Returns
    -------
    Wavepacket[_NS0Inv, _NS1Inv, _BC0Inv]
        Normalized wavepacket
    """
    converted = convert_eigenstate_to_position_basis(get_eigenstate(wavepacket, 0))
    max_idx = np.argmax(np.abs(converted["vector"]), axis=-1)
    max_idx = wrap_index_around_origin_x01(converted["basis"], max_idx)
    mirror_idx = get_x01_mirrored_index(converted["basis"], max_idx)

    bloch_phases_max = _get_bloch_phases(wavepacket, max_idx)
    global_phases_max = _get_global_phases(wavepacket, max_idx)

    bloch_phases_mirror = _get_bloch_phases(wavepacket, mirror_idx)
    global_phases_mirror = _get_global_phases(wavepacket, mirror_idx)

    phi_0 = bloch_phases_max + global_phases_max
    phi_1 = bloch_phases_mirror + global_phases_mirror

    # Normalize such that the average distance phi_0, phi_1 is zero
    averaged_fix = (
        phi_0 + phi_1 - phi_1 + 0.5 * (_wrap_phases(phi_1 - phi_0, np.pi / 2))
    )

    phases = np.exp(-1j * (averaged_fix - angle))
    # Use this to test the convergence of both points
    # ! converted_wavepacket = convert_wavepacket_to_position_basis(wavepacket)
    # ! converted_normalized = converted_wavepacket["vectors"] * phases[:, :, np.newaxis]
    # ! util = BasisConfigUtil(converted_wavepacket["basis"])
    # ! mirrored_idx = util.get_flat_index(mirror_idx, mode="wrap")
    # ! p r int(mirror_idx, max_idx)
    # ! p r int(
    # !     np.average(
    # !         np.abs(
    # !             wrap_phases(
    # !                 np.angle(
    # !                     np.exp(1j * (global_phases_mirror))
    # !                     * converted_normalized[:, :, mirrored_idx]
    # !                 ),
    # !                 np.pi / 2,
    # !             )
    # !         )
    # !     )
    # ! )
    # ! p r int(
    # !     np.average(
    # !         np.abs(
    # !             np.angle(
    # !                 np.exp(1j * global_phases_max)
    # !                 * converted_normalized[
    # !                     :, :, util.get_flat_index(max_idx, mode="wrap")
    # !                 ]
    # !             )
    # !         )
    # !     )
    # ! )
    fixed_eigenvectors = wavepacket["vectors"] * phases[:, :, np.newaxis]

    return {
        "basis": wavepacket["basis"],
        "vectors": fixed_eigenvectors,
        "energies": wavepacket["energies"],
    }


@timed
def normalize_wavepacket_single_point(
    wavepacket: Wavepacket[_NS0Inv, _NS1Inv, _BC0Inv],
    angle: float = 0,
) -> Wavepacket[_NS0Inv, _NS1Inv, _BC0Inv]:
    """
    Normalize a wavepacket using a 'single-point' calculation.

    This makes use of the x0=x1 symmetry in the wavepacket, and the fact that the jump in phase over the symmetry is 0 or pi.
    We therefore only need to normalize one of the maximum points, and we get the other for free

    Parameters
    ----------
    wavepacket : Wavepacket[_NS0Inv, _NS1Inv, _BC0Inv]
        Initial wavepacket
    angle : float, optional
        angle at the origin, by default 0

    Returns
    -------
    Wavepacket[_NS0Inv, _NS1Inv, _BC0Inv]
        Normalized wavepacket
    """
    converted = convert_eigenstate_to_position_basis(get_eigenstate(wavepacket, 0))
    max_idx = np.argmax(np.abs(converted["vector"]), axis=-1)
    max_idx = wrap_index_around_origin_x01(converted["basis"], max_idx)

    bloch_phases = _get_bloch_phases(wavepacket, max_idx)
    global_phases = _get_global_phases(wavepacket, max_idx)

    phases = np.exp(-1j * (bloch_phases + global_phases - angle))
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
