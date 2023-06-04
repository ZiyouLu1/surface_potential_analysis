from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Literal, TypedDict, TypeVar

import numpy as np
from surface_potential_analysis.basis.conversion import (
    basis3d_as_single_point_basis,
)
from surface_potential_analysis.basis.util import (
    Basis3dUtil,
    wrap_index_around_origin_x01,
)
from surface_potential_analysis.wavepacket.wavepacket import (
    get_unfurled_basis,
)

from surface_dynamics_simulation.tunnelling_matrix.util import (
    get_all_single_site_tunnelling_vectors,
    get_occupation_per_state,
)
from surface_dynamics_simulation.tunnelling_simulation.simulation import (
    TunnellingEigenstates,
    calculate_tunnelling_eigenstates,
    get_equilibrium_state,
    get_tunnelling_simulation_state,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import (
        Basis3d,
    )

    from surface_dynamics_simulation.tunnelling_matrix.tunnelling_matrix import (
        TunnellingMatrix,
    )

    from .tunnelling_simulation_state import (
        TunnellingSimulationState,
    )

    _S0Inv = TypeVar("_S0Inv", bound=tuple[int, int, Literal[6]])
    _L0Inv = TypeVar("_L0Inv", bound=int)
    _L1Inv = TypeVar("_L1Inv", bound=int)
    _L2Inv = TypeVar("_L2Inv", bound=int)
    _B3d0Inv = TypeVar("_B3d0Inv", bound=Basis3d[Any, Any, Any])

_N0Inv = TypeVar("_N0Inv", bound=int)


class ISF(TypedDict, Generic[_N0Inv]):
    """Represents the ISF of a tunnelling simulation."""

    times: np.ndarray[tuple[_N0Inv], np.dtype[np.float_]]
    vector: np.ndarray[tuple[_N0Inv], np.dtype[np.float_]]
    """
    Vector representing the ISF calculated at each time t
    """


def _calculate_mean_locations(
    shape: tuple[_L0Inv, _L1Inv, Literal[6]], basis: _B3d0Inv
) -> np.ndarray[tuple[Literal[3], _L0Inv, _L1Inv, Literal[6]], np.dtype[np.float_]]:
    hopping_basis = get_unfurled_basis(
        basis3d_as_single_point_basis(basis),
        (shape[0], shape[1]),
    )
    util = Basis3dUtil(basis)

    nx_points = Basis3dUtil(hopping_basis).nx_points
    nx_points_wrapped = wrap_index_around_origin_x01(hopping_basis, nx_points)
    ffc_locations = util.get_x_points_at_index(nx_points_wrapped)

    locations = np.tile(ffc_locations, (1, shape[2])).reshape(3, *shape)
    hcp_offset = 1 / 3 * (util.delta_x0 + util.delta_x1)
    locations[:, :, :, [1, 4, 5]] += hcp_offset[:, np.newaxis, np.newaxis, np.newaxis]
    return locations  # type: ignore[no-any-return]


def _calculate_isf_from_simulation_state(
    state: TunnellingSimulationState[_N0Inv, _S0Inv],
    basis: _B3d0Inv,
    dk: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]],
) -> ISF[_N0Inv]:
    """
    Given an concrete initial simulation state, calculate the ISF.

    Parameters
    ----------
    state : TunnellingSimulationState[_N0Inv, _S0Inv]
        intital simulation state
    basis : _B3d0Inv
    dk : np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]

    Returns
    -------
    ISF[_N0Inv]
    """
    mean_locations = _calculate_mean_locations(state["shape"], basis).reshape(3, -1)  # type: ignore[arg-type,var-annotated]
    initial_location = np.average(
        mean_locations, axis=1, weights=state["vectors"][:, 0]
    )
    distances = mean_locations - initial_location[:, np.newaxis]

    mean_phi = np.tensordot(dk, distances, axes=(0, 0))
    vector = np.tensordot(np.exp(-1j * mean_phi), state["vectors"], axes=(0, 0))

    relative_times = state["times"] - state["times"][0]
    return {"vector": vector, "times": relative_times}


def _get_occupation_per_state_at_equilibrium(
    eigenstates: TunnellingEigenstates[tuple[_L0Inv, _L1Inv, _L2Inv]]
) -> np.ndarray[tuple[_L2Inv], np.dtype[np.float_]]:
    equilibrium_state = get_equilibrium_state(eigenstates)
    return get_occupation_per_state(equilibrium_state)


def get_isf(
    eigenstates: TunnellingEigenstates[tuple[_L0Inv, _L1Inv, Literal[6]]],
    basis: _B3d0Inv,
    dk: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]],
    times: np.ndarray[tuple[_N0Inv], np.dtype[np.float_]],
) -> ISF[_N0Inv]:
    """
    Get the ISF, averaged over all initial configurations of the surface.

    Parameters
    ----------
    matrix : TunnellingMatrix[_S0Inv]
    basis : _B3d0Inv
    dk : np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]

    Returns
    -------
    ISF[_N0Inv]
    """
    vectors = get_all_single_site_tunnelling_vectors(eigenstates["shape"])
    occupations = _get_occupation_per_state_at_equilibrium(eigenstates)

    isf_vectors = np.zeros((len(vectors), times.size))
    for i, vector in enumerate(vectors):
        state = get_tunnelling_simulation_state(eigenstates, vector, times)
        isf = _calculate_isf_from_simulation_state(state, basis, dk)
        isf_vectors[i] = isf["vector"]

    average_isf_vector = np.average(isf_vectors, axis=0, weights=occupations)
    return {"times": times, "vector": average_isf_vector}


def calculate_isf(
    matrix: TunnellingMatrix[tuple[_L0Inv, _L1Inv, Literal[6]]],
    basis: _B3d0Inv,
    dk: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]],
    times: np.ndarray[tuple[_N0Inv], np.dtype[np.float_]],
) -> ISF[_N0Inv]:
    """
    Calculate the ISF, averaged over all initial configurations of the surface.

    Parameters
    ----------
    matrix : TunnellingMatrix[_S0Inv]
    basis : _B3d0Inv
    dk : np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]

    Returns
    -------
    ISF[_N0Inv]
    """
    eigenstates = calculate_tunnelling_eigenstates(matrix)
    return get_isf(eigenstates, basis, dk, times)
