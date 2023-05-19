from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Literal, TypedDict, TypeVar

import numpy as np
from surface_potential_analysis.basis_config.basis_config import (
    BasisConfig,
    BasisConfigUtil,
    get_single_point_basis_in,
    wrap_index_around_origin_x01,
)
from surface_potential_analysis.wavepacket.eigenstate_conversion import (
    get_unfurled_basis,
)

if TYPE_CHECKING:
    from .tunnelling_simulation_state import (
        TunnellingSimulationState,
    )

    _S0Inv = TypeVar("_S0Inv", bound=tuple[int, int, Literal[6]])
    _N0Inv = TypeVar("_N0Inv", bound=int)
    _L0Inv = TypeVar("_L0Inv", bound=int)
    _L1Inv = TypeVar("_L1Inv", bound=int)
    _BC0Inv = TypeVar("_BC0Inv", bound=BasisConfig[Any, Any, Any])


def _calculate_mean_locations(
    shape: tuple[_L0Inv, _L1Inv, Literal[6]], basis: _BC0Inv
) -> np.ndarray[tuple[Literal[3], _L0Inv, _L1Inv, Literal[6]], np.dtype[np.float_]]:
    hopping_basis = get_unfurled_basis(
        get_single_point_basis_in(basis, "position"), (shape[0], shape[1])
    )
    util = BasisConfigUtil(basis)

    nx_points = BasisConfigUtil(hopping_basis).nx_points
    nx_points_wrapped = wrap_index_around_origin_x01(hopping_basis, nx_points)
    ffc_locations = util.get_x_points_at_index(nx_points_wrapped)

    locations = np.tile(ffc_locations, (1, shape[2])).reshape(3, *shape)
    hcp_offset = 1 / 3 * (util.delta_x0 + util.delta_x1)
    locations[:, :, :, [1, 4, 5]] += hcp_offset[:, np.newaxis, np.newaxis, np.newaxis]
    return locations  # type: ignore[no-any-return]


class ISF(TypedDict, Generic[_N0Inv]):
    """Represents the ISF of a tunnelling simulation."""

    times: np.ndarray[tuple[_N0Inv], np.dtype[np.float_]]
    vector: np.ndarray[tuple[_N0Inv], np.dtype[np.float_]]
    """
    Vector representing the ISF calculated at each time t
    """


def _calculate_isf_from_simulation_state(
    state: TunnellingSimulationState[_N0Inv, _S0Inv],
    basis: _BC0Inv,
    dk: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]],
) -> ISF[_N0Inv]:
    """
    Given an concrete initial simulation state, calculate the ISF.

    Parameters
    ----------
    state : TunnellingSimulationState[_N0Inv, _S0Inv]
        intital simulation state
    basis : _BC0Inv
    dk : np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]

    Returns
    -------
    ISF[_N0Inv]
    """
    relative_times = state["times"] - state["times"][0]
    mean_locations = _calculate_mean_locations(state["shape"], basis).reshape(3, -1)  # type: ignore[arg-type,var-annotated]
    initial_location = np.average(
        mean_locations, axis=1, weights=state["vectors"][:, 0]
    )
    distances = mean_locations - initial_location[:, np.newaxis]
    mean_phi = np.tensordot(dk, distances, axes=0)
    vector = np.sum(np.exp(-1j * mean_phi) * state["vectors"], axis=0)
    return {"vector": vector, "times": relative_times}
