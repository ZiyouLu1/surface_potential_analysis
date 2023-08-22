from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from surface_potential_analysis.basis.util import (
    BasisUtil,
    wrap_index_around_origin,
)
from surface_potential_analysis.state_vector.conversion import (
    convert_state_dual_vector_to_basis,
    convert_state_vector_to_momentum_basis,
    convert_state_vector_to_position_basis,
)
from surface_potential_analysis.state_vector.state_vector import (
    StateVector,
    as_dual_vector,
    calculate_inner_product,
)
from surface_potential_analysis.wavepacket.get_eigenstate import (
    get_all_eigenstates,
    get_eigenstate,
    get_tight_binding_state,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from surface_potential_analysis._types import (
        SingleIndexLike,
        SingleStackedIndexLike,
    )
    from surface_potential_analysis.axis.axis import FundamentalPositionAxis
    from surface_potential_analysis.basis.basis import AxisWithLengthBasis
    from surface_potential_analysis.wavepacket.wavepacket import (
        Wavepacket,
    )

    _B0Inv = TypeVar("_B0Inv", bound=AxisWithLengthBasis[Any])
    _B1Inv = TypeVar("_B1Inv", bound=AxisWithLengthBasis[Any])

    _S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])


def get_projection_coefficients(
    projection: StateVector[_B0Inv], states: Sequence[StateVector[_B1Inv]]
) -> np.ndarray[tuple[int], np.dtype[np.complex_]]:
    """
    Given a projection and a set of states, calculate the coefficients.

    Parameters
    ----------
    projection : StateVector[_B0Inv]
    states : Sequence[StateVector[_B1Inv]]

    Returns
    -------
    np.ndarray[tuple[int], np.dtype[np.complex_]]
    """
    coefficients = np.zeros(len(states), dtype=np.complex_)
    projection_dual = as_dual_vector(projection)
    for i, state in enumerate(states):
        converted = convert_state_dual_vector_to_basis(projection_dual, state["basis"])
        coefficients[i] = calculate_inner_product(state, converted)

    return coefficients  # type: ignore[no-any-return]


def _get_normalized_projection_coefficients(
    projection: StateVector[_B0Inv], states: Sequence[StateVector[_B1Inv]]
) -> np.ndarray[tuple[int], np.dtype[np.complex_]]:
    coefficients = get_projection_coefficients(projection, states)
    # Normalize the coefficients.
    # This means that the product between the projection and the eigenstates
    # all have the same phase
    return coefficients / np.abs(coefficients)  # type: ignore[no-any-return]


def localize_wavepacket_projection(
    wavepacket: Wavepacket[_S0Inv, _B0Inv],
    projection: StateVector[_B1Inv],
) -> Wavepacket[_S0Inv, _B0Inv]:
    """
    Given a wavepacket, localize using the given projection.

    Parameters
    ----------
    wavepacket : Wavepacket[_S0Inv, _B0Inv]
    projection : StateVector[_B1Inv]

    Returns
    -------
    Wavepacket[_S0Inv, _B0Inv]
    """
    coefficients = _get_normalized_projection_coefficients(
        projection, get_all_eigenstates(wavepacket)
    )

    return {
        "basis": wavepacket["basis"],
        "eigenvalues": wavepacket["eigenvalues"],
        "shape": wavepacket["shape"],
        "vectors": wavepacket["vectors"] / coefficients[:, np.newaxis],
    }


def localize_tight_binding_projection(
    wavepacket: Wavepacket[_S0Inv, _B0Inv]
) -> Wavepacket[_S0Inv, _B0Inv]:
    """
    Given a wavepacket, localize using a tight binding projection.

    Parameters
    ----------
    wavepacket : Wavepacket[_S0Inv, _B0Inv]

    Returns
    -------
    Wavepacket[_S0Inv, _B0Inv]
    """
    # Initial guess is that the localized state is just the state of some eigenstate
    # truncated at the edge of the first
    # unit cell, centered at the two point max of the wavefunction
    projection = get_tight_binding_state(wavepacket)
    # Better performace if we provide the projection in transformed basis
    transformed = convert_state_vector_to_momentum_basis(projection)
    return localize_wavepacket_projection(wavepacket, transformed)


def _get_single_point_state(
    wavepacket: Wavepacket[_S0Inv, _B0Inv],
    idx: SingleIndexLike = 0,
    origin: SingleStackedIndexLike | None = None,
) -> StateVector[tuple[FundamentalPositionAxis[Any, Any], ...]]:
    state_0 = convert_state_vector_to_position_basis(get_eigenstate(wavepacket, idx))
    util = BasisUtil(state_0["basis"])
    if origin is None:
        idx_0: SingleStackedIndexLike = util.get_stacked_index(
            np.argmax(np.abs(state_0["vector"]), axis=-1)
        )
        origin = wrap_index_around_origin(wavepacket["basis"], idx_0, (0, 0, 0), (0, 1))

    out: StateVector[tuple[FundamentalPositionAxis[Any, Any], ...]] = {
        "basis": state_0["basis"],
        "vector": np.zeros_like(state_0["vector"]),
    }
    out["vector"][util.get_flat_index(origin, mode="wrap")] = 1
    return out


def localize_single_point_projection(
    wavepacket: Wavepacket[_S0Inv, _B0Inv]
) -> Wavepacket[_S0Inv, _B0Inv]:
    """
    Given a wavepacket, localize using a tight binding projection.

    Parameters
    ----------
    wavepacket : Wavepacket[_S0Inv, _B0Inv]

    Returns
    -------
    Wavepacket[_S0Inv, _B0Inv]
    """
    # Initial guess is that the localized state is just the state of some eigenstate
    # truncated at the edge of the first
    # unit cell, centered at the two point max of the wavefunction
    projection = _get_single_point_state(wavepacket)
    # Will have better performace if we provide it in a truncated position basis
    return localize_wavepacket_projection(wavepacket, projection)


def get_exponential_state(
    wavepacket: Wavepacket[_S0Inv, _B0Inv]
) -> StateVector[tuple[FundamentalPositionAxis[Any, Any], ...]]:
    """
    Given a wavepacket, get the state decaying exponentially from the maximum.

    Parameters
    ----------
    wavepacket : Wavepacket[_S0Inv, _B0Inv]
        The initial wavepacket
    idx : SingleIndexLike, optional
        The index of the state vector to use as reference, by default 0
    origin : SingleIndexLike | None, optional
        The origin about which to produce the localized state, by default the maximum of the wavefunction

    Returns
    -------
    StateVector[tuple[FundamentalPositionAxis[Any, Any], ...]]
        The localized state under the tight binding approximation
    """
    state_0 = convert_state_vector_to_position_basis(get_eigenstate(wavepacket, 0))

    util = BasisUtil(state_0["basis"])
    idx_0: SingleStackedIndexLike = util.get_stacked_index(
        np.argmax(np.abs(state_0["vector"]), axis=-1)
    )
    origin = wrap_index_around_origin(wavepacket["basis"], idx_0, (0, 0, 0), (0, 1))

    coordinates = wrap_index_around_origin(state_0["basis"], util.nx_points, origin)
    unit_cell_util = BasisUtil(wavepacket["basis"])
    dx0 = coordinates[0] - origin[0] / unit_cell_util.fundamental_shape[0]
    dx1 = coordinates[1] - origin[1] / unit_cell_util.fundamental_shape[1]
    dx2 = coordinates[2] - origin[2] / unit_cell_util.fundamental_shape[2]

    out: StateVector[tuple[FundamentalPositionAxis[Any, Any], ...]] = {
        "basis": state_0["basis"],
        "vector": np.zeros_like(state_0["vector"]),
    }
    out["vector"] = np.exp(-(dx0**2 + dx1**2 + dx2**2))
    out["vector"] /= np.linalg.norm(out["vector"])
    return out


def _get_exponential_decay_state(
    wavepacket: Wavepacket[_S0Inv, _B0Inv]
) -> StateVector[tuple[FundamentalPositionAxis[Any, Any], ...]]:
    exponential = get_exponential_state(wavepacket)
    tight_binding = convert_state_vector_to_position_basis(
        get_eigenstate(wavepacket, 0)
    )
    out: StateVector[tuple[FundamentalPositionAxis[Any, Any], ...]] = {
        "basis": exponential["basis"],
        "vector": exponential["vector"] * tight_binding["vector"],
    }
    out["vector"] /= np.linalg.norm(out["vector"])
    return out


def localize_exponential_decay_projection(
    wavepacket: Wavepacket[_S0Inv, _B0Inv]
) -> Wavepacket[_S0Inv, _B0Inv]:
    """
    Given a wavepacket, localize using a tight binding projection.

    Parameters
    ----------
    wavepacket : Wavepacket[_S0Inv, _B0Inv]

    Returns
    -------
    Wavepacket[_S0Inv, _B0Inv]
    """
    # Initial guess is that the localized state is the tight binding state
    # multiplied by an exponential
    projection = _get_exponential_decay_state(wavepacket)
    return localize_wavepacket_projection(wavepacket, projection)
