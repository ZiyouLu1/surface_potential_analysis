from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from surface_potential_analysis.axis.axis import (
    FundamentalBasis,
)
from surface_potential_analysis.axis.stacked_axis import StackedBasis
from surface_potential_analysis.axis.util import (
    BasisUtil,
)
from surface_potential_analysis.operator.conversion import (
    convert_operator_to_basis,
)
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_position_basis,
)
from surface_potential_analysis.state_vector.conversion import (
    convert_state_vector_to_basis,
)
from surface_potential_analysis.state_vector.eigenstate_calculation import (
    calculate_eigenvectors_hermitian,
    calculate_operator_inner_product,
)
from surface_potential_analysis.state_vector.state_vector import (
    StateVector,
    as_dual_vector,
)
from surface_potential_analysis.util.decorators import timed
from surface_potential_analysis.wavepacket.eigenstate_conversion import (
    unfurl_wavepacket,
)
from surface_potential_analysis.wavepacket.get_eigenstate import get_all_eigenstates
from surface_potential_analysis.wavepacket.wavepacket import (
    WavepacketWithEigenvalues,
    get_unfurled_basis,
)

if TYPE_CHECKING:
    from surface_potential_analysis.axis.stacked_axis import (
        StackedBasisLike,
    )
    from surface_potential_analysis.operator.operator import SingleBasisOperator

    _B1Inv = TypeVar("_B1Inv", bound=StackedBasisLike[*tuple[Any, ...]])
    _B2Inv = TypeVar("_B2Inv", bound=StackedBasisLike[*tuple[Any, ...]])
    _B0Inv = TypeVar("_B0Inv", bound=StackedBasisLike[*tuple[Any, ...]])


def _get_position_operator(basis: _B1Inv) -> SingleBasisOperator[_B1Inv]:
    util = BasisUtil(basis)
    # We only get the location in the x0 direction here
    locations = util.x_points_stacked[0]

    basis_position = stacked_basis_as_fundamental_position_basis(basis)
    operator: SingleBasisOperator[Any] = {
        "basis": StackedBasis(basis_position, basis_position),
        "data": np.diag(locations),
    }
    return convert_operator_to_basis(operator, StackedBasis(basis, basis))


@timed
def _get_operator_between_states(
    states: list[StateVector[_B1Inv]], operator: SingleBasisOperator[_B1Inv]
) -> SingleBasisOperator[FundamentalBasis[Any]]:
    n_states = len(states)
    array = np.zeros((n_states, n_states), dtype=np.complex_)
    for i in range(n_states):
        dual_vector = as_dual_vector(states[i])
        for j in range(n_states):
            vector = states[j]
            array[i, j] = calculate_operator_inner_product(
                dual_vector, operator, vector
            )

    basis = FundamentalBasis(n_states)
    return {"data": array, "basis": StackedBasis(basis, basis)}


def _localize_operator(
    wavepacket: WavepacketWithEigenvalues[_B0Inv, _B1Inv],
    operator: SingleBasisOperator[_B2Inv],
) -> list[WavepacketWithEigenvalues[_B0Inv, _B1Inv]]:
    states = [
        convert_state_vector_to_basis(state, operator["basis"][0])
        for state in get_all_eigenstates(wavepacket)
    ]
    operator_between_states = _get_operator_between_states(states, operator)
    eigenstates = calculate_eigenvectors_hermitian(operator_between_states)
    return [
        {
            "basis": wavepacket["basis"],
            "eigenvalues": wavepacket["eigenvalues"],
            "data": wavepacket["data"] * vector[:, np.newaxis],
        }
        for vector in eigenstates["data"]
    ]


def localize_position_operator(
    wavepacket: WavepacketWithEigenvalues[_B0Inv, _B1Inv]
) -> list[WavepacketWithEigenvalues[_B0Inv, _B1Inv]]:
    """
    Given a wavepacket generate a set of normalized wavepackets using the operator method.

    Parameters
    ----------
    wavepacket : Wavepacket[_S0Inv, _B0Inv]

    Returns
    -------
    list[Wavepacket[_S0Inv, _B0Inv]]
    """
    basis = stacked_basis_as_fundamental_position_basis(
        get_unfurled_basis(wavepacket["basis"])
    )
    operator_position = _get_position_operator(basis)
    return _localize_operator(wavepacket, operator_position)


def localize_position_operator_many_band(
    wavepackets: list[WavepacketWithEigenvalues[_B0Inv, _B1Inv]]
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
    basis = stacked_basis_as_fundamental_position_basis(
        get_unfurled_basis(wavepackets[0]["basis"])
    )
    states = [
        convert_state_vector_to_basis(state, basis)
        for wavepacket in wavepackets
        for state in get_all_eigenstates(wavepacket)
    ]
    operator_position = _get_position_operator(basis)
    operator = _get_operator_between_states(states, operator_position)
    eigenstates = calculate_eigenvectors_hermitian(operator)
    state_vectors = np.array([s["data"] for s in states])
    return [
        {
            "basis": basis,
            "data": np.tensordot(vector, state_vectors, axes=(0, 0)),
        }
        for vector in eigenstates["data"]
    ]


def localize_position_operator_many_band_individual(
    wavepackets: list[WavepacketWithEigenvalues[_B0Inv, _B1Inv]]
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
    list_shape = (wavepackets[0]["basis"][0]).shape
    states = [
        unfurl_wavepacket(
            localize_position_operator(wavepacket)[np.prod(list_shape) // 4]
        )
        for wavepacket in wavepackets
    ]
    operator_position = _get_position_operator(states[0]["basis"])
    operator = _get_operator_between_states(states, operator_position)  # type: ignore[arg-type]
    eigenstates = calculate_eigenvectors_hermitian(operator)
    state_vectors = np.array([s["data"] for s in states])
    return [
        {
            "basis": states[0]["basis"],
            "data": np.tensordot(vector, state_vectors, axes=(0, 0)),
        }
        for vector in eigenstates["data"]
    ]
