from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar, Unpack, overload

import numpy as np

from surface_potential_analysis.axis.axis import FundamentalPositionAxis
from surface_potential_analysis.basis.conversion import (
    basis_as_fundamental_position_basis,
)
from surface_potential_analysis.basis.util import (
    Basis3dUtil,
    BasisUtil,
    get_x01_mirrored_index,
    wrap_index_around_origin_x01,
)
from surface_potential_analysis.operator.conversion import (
    convert_operator_to_basis,
)
from surface_potential_analysis.state_vector.conversion import (
    convert_state_vector_to_basis,
    convert_state_vector_to_position_basis,
)
from surface_potential_analysis.state_vector.eigenstate_calculation import (
    calculate_eigenstates_hermitian,
    calculate_inner_product,
)
from surface_potential_analysis.state_vector.state_vector import (
    StateVector,
    as_dual_vector,
)
from surface_potential_analysis.util.decorators import timed
from surface_potential_analysis.wavepacket.conversion import (
    convert_wavepacket_to_position_basis,
)
from surface_potential_analysis.wavepacket.eigenstate_conversion import (
    unfurl_wavepacket,
)

from .get_eigenstate import get_eigenstate, get_eigenstates
from .wavepacket import (
    Wavepacket,
    Wavepacket3dWith2dSamples,
    get_unfurled_basis,
    get_wavepacket_sample_fractions,
)

if TYPE_CHECKING:
    from surface_potential_analysis._types import (
        ArrayIndexLike,
        SingleIndexLike,
        SingleStackedIndexLike3d,
    )
    from surface_potential_analysis.basis.basis import Basis, Basis3d
    from surface_potential_analysis.operator.operator import SingleBasisOperator

    _B3d0Inv = TypeVar("_B3d0Inv", bound=Basis3d[Any, Any, Any])
    _B0Inv = TypeVar("_B0Inv", bound=Basis[Any])
    _B1Inv = TypeVar("_B1Inv", bound=Basis[Any])

    _NS0Inv = TypeVar("_NS0Inv", bound=int)
    _NS1Inv = TypeVar("_NS1Inv", bound=int)

    _S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])
    _S1Inv = TypeVar("_S1Inv", bound=tuple[int, ...])

    _WInv = TypeVar("_WInv", bound=Wavepacket[Any, Any])


@overload
def _get_global_phases(
    wavepacket: _WInv,
    idx: SingleIndexLike,
) -> np.ndarray[tuple[int], np.dtype[np.float_]]:
    ...


@overload
def _get_global_phases(
    wavepacket: _WInv,
    idx: ArrayIndexLike[_S1Inv],
) -> np.ndarray[tuple[int, Unpack[_S1Inv]], np.dtype[np.float_]]:
    ...


def _get_global_phases(  # type: ignore[misc]
    wavepacket: _WInv,
    idx: SingleIndexLike | ArrayIndexLike[_S1Inv],
) -> (
    np.ndarray[tuple[int, Unpack[_S1Inv]], np.dtype[np.float_]]
    | np.ndarray[int, np.dtype[np.float_]]
):
    """
    Get the global bloch phase at a given index in the irreducible cell.

    returns a list of the global phase of each eigenstate in the wavepacket

    Parameters
    ----------
    wavepacket : Wavepacket[_NS0Inv, _NS1Inv, Basis3d[_A3d0Inv, _A3d1Inv, _A3d2Inv]]
        The wavepacket to get the global phase for
    idx : int | tuple[int, int, int], optional
        The index in ravelled or unravelled form, by default 0

    Returns
    -------
    np.ndarray[tuple[int], np.dtype[np.float_]]
        phases for each sample in the wavepacket
    """
    basis = basis_as_fundamental_position_basis(wavepacket["basis"])  # type: ignore[arg-type,var-annotated]
    util = BasisUtil(basis)

    nx_points = idx if isinstance(idx, tuple) else util.get_stacked_index(idx)
    nx_fractions = tuple(a / ni for (a, ni) in zip(nx_points, util.shape, strict=True))

    nk_fractions = get_wavepacket_sample_fractions(wavepacket["shape"])
    return 2 * np.pi * np.tensordot(nk_fractions, nx_fractions, axes=(0, 0))  # type: ignore[no-any-return]


def _get_bloch_wavefunction_phases(
    wavepacket: _WInv,
    idx: SingleIndexLike = 0,
) -> np.ndarray[tuple[int], np.dtype[np.float_]]:
    """
    Get the phase of the bloch wavefunctions at the given point in position space.

    Parameters
    ----------
    wavepacket : Wavepacket[ _NS0Inv, _NS1Inv, Basis3d[PositionBasis[_L0Inv], PositionBasis[_L1Inv], _A3d2Inv]]
        the wavepacket to calculate the phase of
    idx : SingleIndexLike, optional
        the index in real space, by default 0

    Returns
    -------
    np.ndarray[tuple[_NS0Inv, _NS1Inv], np.dtype[np.float_]]
        the angle for each point in the wavepacket
    """
    converted = convert_wavepacket_to_position_basis(wavepacket)
    util = BasisUtil(converted["basis"])
    idx = util.get_flat_index(idx, mode="wrap") if isinstance(idx, tuple) else idx

    return np.angle(converted["vectors"][:, idx])  # type: ignore[return-value]


@timed
def localize_tightly_bound_wavepacket_idx(
    wavepacket: _WInv,
    idx: SingleIndexLike = 0,
    angle: float = 0,
) -> _WInv:
    """
    localize a wavepacket in momentum basis.

    Parameters
    ----------
    wavepacket : Wavepacket[ _NS0Inv, _NS1Inv, Basis3d[ TruncatedBasis[_L0Inv, MomentumBasis[_LF0Inv]], TruncatedBasis[_L1Inv, MomentumBasis[_LF1Inv]], ExplicitBasis[_L2Inv, PositionBasis[_LF2Inv]], ], ]
    idx : SingleIndexLike , optional
        Index of the eigenstate to localize, by default 0
        This index is taken in the irreducible unit cell
    angle: float, optional
        Angle to localize the wavepacket to at the point idx.
        Each wavefunction will have the phase exp(i * angle) at the position
        given by idx

    Returns
    -------
    Wavepacket[ _NS0Inv, _NS1Inv, Basis3d[ TruncatedBasis[_L0Inv, MomentumBasis[_LF0Inv]], TruncatedBasis[_L1Inv, MomentumBasis[_LF1Inv]], ExplicitBasis[_L2Inv, PositionBasis[_LF2Inv]], ], ]
    """
    bloch_angles = _get_bloch_wavefunction_phases(wavepacket, idx)
    global_phases = _get_global_phases(wavepacket, idx)

    phases = np.exp(-1j * (bloch_angles + global_phases - angle))
    fixed_eigenvectors = wavepacket["vectors"] * phases[:, np.newaxis]

    return {
        "basis": wavepacket["basis"],
        "shape": wavepacket["shape"],
        "vectors": fixed_eigenvectors,
        "energies": wavepacket["energies"],
    }


def get_wavepacket_two_points(
    wavepacket: Wavepacket3dWith2dSamples[_NS0Inv, _NS1Inv, _B3d0Inv],
    offset: tuple[int, int] = (0, 0),
) -> tuple[SingleStackedIndexLike3d, SingleStackedIndexLike3d]:
    """
    Get the index of the maximum, and the index mirrored in x01 for the wavepacket, wrapped about the given offset.

    Parameters
    ----------
    wavepacket : Wavepacket[_NS0Inv, _NS1Inv, _B3d0Inv]
    offset : tuple[int, int], optional
        offset of the point to use as an origin, by default (0, 0)

    Returns
    -------
    tuple[SingleStackedIndexLike, SingleStackedIndexLike]
    """
    util = Basis3dUtil(wavepacket["basis"])
    origin = (util.shape[0] * offset[0], util.shape[1] * offset[1], 0)

    converted = convert_state_vector_to_position_basis(get_eigenstate(wavepacket, 0))  # type: ignore[arg-type,var-annotated]
    idx_0: SingleStackedIndexLike3d = np.argmax(np.abs(converted["vector"]), axis=-1)
    idx_0 = wrap_index_around_origin_x01(converted["basis"], idx_0, origin)
    idx_1 = get_x01_mirrored_index(converted["basis"], idx_0)
    idx_1 = wrap_index_around_origin_x01(converted["basis"], idx_1, origin)
    return (idx_0, idx_1)


def _wrap_phases(
    phases: np.ndarray[tuple[int], np.dtype[np.float_]], half_width: float = np.pi
) -> np.ndarray[tuple[int], np.dtype[np.float_]]:
    return (phases + half_width) % (2 * half_width) - half_width  # type: ignore[no-any-return]


@timed
def localize_tightly_bound_wavepacket_two_point_max(
    wavepacket: Wavepacket3dWith2dSamples[_NS0Inv, _NS1Inv, _B3d0Inv],
    offset: tuple[int, int] = (0, 0),
    angle: float = 0,
) -> Wavepacket3dWith2dSamples[_NS0Inv, _NS1Inv, _B3d0Inv]:
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
    wavepacket : Wavepacket[_NS0Inv, _NS1Inv, _B3d0Inv]
        Initial wavepacket
    offset: tuple[int, int], optional
        offset of the point to localize, by default (0,0)
    angle : float, optional
        angle at the origin, by default 0

    Returns
    -------
    Wavepacket[_NS0Inv, _NS1Inv, _B3d0Inv]
        Normalized wavepacket
    """
    idx_0, idx_1 = get_wavepacket_two_points(wavepacket, offset)

    bloch_phases_0 = _get_bloch_wavefunction_phases(wavepacket, idx_0)
    global_phases_0 = _get_global_phases(wavepacket, idx_0)
    phi_0 = bloch_phases_0 + global_phases_0

    bloch_phases_1 = _get_bloch_wavefunction_phases(wavepacket, idx_1)
    global_phases_1 = _get_global_phases(wavepacket, idx_1)
    phi_1 = bloch_phases_1 + global_phases_1

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
        "shape": wavepacket["shape"],
        "vectors": fixed_eigenvectors,
        "energies": wavepacket["energies"],
    }


@timed
def localize_tightly_bound_wavepacket_max_point(
    wavepacket: _WInv,
    angle: float = 0,
) -> _WInv:
    """
    Normalize a wavepacket using a 'single-point' calculation.

    This makes use of the x0=x1 symmetry in the wavepacket, and the fact that the jump in phase over the symmetry is 0 or pi.
    We therefore only need to localize one of the maximum points, and we get the other for free

    Parameters
    ----------
    wavepacket : Wavepacket[_NS0Inv, _NS1Inv, _B3d0Inv]
        Initial wavepacket
    angle : float, optional
        angle at the origin, by default 0

    Returns
    -------
    Wavepacket[_NS0Inv, _NS1Inv, _B3d0Inv]
        Normalized wavepacket
    """
    converted = convert_state_vector_to_position_basis(get_eigenstate(wavepacket, 0))  # type: ignore[arg-type,var-annotated]
    max_idx = np.argmax(np.abs(converted["vector"]), axis=-1)
    max_idx = wrap_index_around_origin_x01(converted["basis"], max_idx)

    bloch_phases = _get_bloch_wavefunction_phases(wavepacket, max_idx)
    global_phases = _get_global_phases(wavepacket, max_idx)

    phases = np.exp(-1j * (bloch_phases + global_phases - angle))
    fixed_eigenvectors = wavepacket["vectors"] * phases[:, :, np.newaxis]

    return {  # type: ignore[return-value]
        "basis": wavepacket["basis"],
        "vectors": fixed_eigenvectors,
        "energies": wavepacket["energies"],
        "shape": wavepacket["shape"],
    }


def _get_position_operator(basis: _B0Inv) -> SingleBasisOperator[_B0Inv]:
    util = BasisUtil(basis)
    # We only get the location in the x0 direction here
    locations = util.x_points[0]

    basis_position = basis_as_fundamental_position_basis(basis)
    operator: SingleBasisOperator[Any] = {
        "basis": basis_position,
        "dual_basis": basis_position,
        "array": np.diag(locations),
    }
    return convert_operator_to_basis(operator, basis, basis)


@timed
def _get_operator_between_states(
    states: list[StateVector[_B0Inv]], operator: SingleBasisOperator[_B0Inv]
) -> SingleBasisOperator[tuple[FundamentalPositionAxis[Any, Literal[1]]]]:
    n_states = len(states)
    array = np.zeros((n_states, n_states), dtype=np.complex_)
    for i in range(n_states):
        dual_vector = as_dual_vector(states[i])
        for j in range(n_states):
            vector = states[j]
            array[i, j] = calculate_inner_product(dual_vector, operator, vector)

    basis = (FundamentalPositionAxis(np.array([0]), n_states),)
    return {"array": array, "basis": basis, "dual_basis": basis}


def _localize_operator(
    wavepacket: Wavepacket[_S0Inv, _B0Inv], operator: SingleBasisOperator[_B1Inv]
) -> list[Wavepacket[_S0Inv, _B0Inv]]:
    states = [
        convert_state_vector_to_basis(state, operator["basis"])
        for state in get_eigenstates(wavepacket)
    ]
    operator_between_states = _get_operator_between_states(states, operator)
    eigenstates = calculate_eigenstates_hermitian(operator_between_states)
    return [
        {
            "basis": wavepacket["basis"],
            "energies": wavepacket["energies"],
            "shape": wavepacket["shape"],
            "vectors": wavepacket["vectors"] * vector[:, np.newaxis],
        }
        for vector in eigenstates["vectors"]
    ]


def localize_position_operator(
    wavepacket: Wavepacket[_S0Inv, _B0Inv]
) -> list[Wavepacket[_S0Inv, _B0Inv]]:
    """
    Given a wavepacket generate a set of normalized wavepackets using the operator method.

    Parameters
    ----------
    wavepacket : Wavepacket[_S0Inv, _B0Inv]

    Returns
    -------
    list[Wavepacket[_S0Inv, _B0Inv]]
    """
    basis = basis_as_fundamental_position_basis(
        get_unfurled_basis(wavepacket["basis"], wavepacket["shape"])
    )
    operator_position = _get_position_operator(basis)
    return _localize_operator(wavepacket, operator_position)


def localize_position_operator_many_band(
    wavepackets: list[Wavepacket[_S0Inv, _B0Inv]]
) -> list[StateVector[Any]]:
    """
    Given a sequence of wavepackets at each band, get all possible eigenstates of position.

    Parameters
    ----------
    wavepackets : list[Wavepacket[_S0Inv, _B0Inv]]

    Returns
    -------
    list[StateVector[Any]]
    """
    basis = basis_as_fundamental_position_basis(
        get_unfurled_basis(wavepackets[0]["basis"], wavepackets[0]["shape"])
    )
    states = [
        convert_state_vector_to_basis(state, basis)
        for wavepacket in wavepackets
        for state in get_eigenstates(wavepacket)
    ]
    operator_position = _get_position_operator(basis)
    operator = _get_operator_between_states(states, operator_position)
    eigenstates = calculate_eigenstates_hermitian(operator)
    state_vectors = np.array([s["vector"] for s in states])
    return [
        {
            "basis": basis,
            "vector": np.tensordot(vector, state_vectors, axes=(0, 0)),
        }
        for vector in eigenstates["vectors"]
    ]


def localize_position_operator_many_band_individual(
    wavepackets: list[Wavepacket[_S0Inv, _B0Inv]]
) -> list[StateVector[Any]]:
    """
    Given a wavepacket generate a set of normalized wavepackets using the operator method.

    Parameters
    ----------
    wavepacket : Wavepacket[_S0Inv, _B0Inv]

    Returns
    -------
    list[Wavepacket[_S0Inv, _B0Inv]]
    """
    states = [
        unfurl_wavepacket(
            localize_position_operator(wavepacket)[np.prod(wavepacket["shape"]) // 4]
        )
        for wavepacket in wavepackets
    ]
    operator_position = _get_position_operator(states[0]["basis"])
    operator = _get_operator_between_states(states, operator_position)
    eigenstates = calculate_eigenstates_hermitian(operator)
    state_vectors = np.array([s["vector"] for s in states])
    return [
        {
            "basis": states[0]["basis"],
            "vector": np.tensordot(vector, state_vectors, axes=(0, 0)),
        }
        for vector in eigenstates["vectors"]
    ]
