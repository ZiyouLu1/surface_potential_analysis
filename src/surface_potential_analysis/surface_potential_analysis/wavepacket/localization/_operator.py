from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np

from surface_potential_analysis.axis.axis import FundamentalPositionAxis
from surface_potential_analysis.basis.conversion import (
    basis_as_fundamental_position_basis,
)
from surface_potential_analysis.basis.util import (
    AxisWithLengthBasisUtil,
    BasisUtil,
)
from surface_potential_analysis.operator.conversion import (
    convert_operator_to_basis,
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
    from surface_potential_analysis.basis.basis import AxisWithLengthBasis, Basis
    from surface_potential_analysis.operator.operator import SingleBasisOperator

    _B1Inv = TypeVar("_B1Inv", bound=AxisWithLengthBasis[Any])
    _B2Inv = TypeVar("_B2Inv", bound=AxisWithLengthBasis[Any])

    _B0Inv = TypeVar("_B0Inv", bound=Basis)


def _get_position_operator(basis: _B1Inv) -> SingleBasisOperator[_B1Inv]:
    util = AxisWithLengthBasisUtil(basis)
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
    states: list[StateVector[_B1Inv]], operator: SingleBasisOperator[_B1Inv]
) -> SingleBasisOperator[tuple[FundamentalPositionAxis[Any, Literal[1]]]]:
    n_states = len(states)
    array = np.zeros((n_states, n_states), dtype=np.complex_)
    for i in range(n_states):
        dual_vector = as_dual_vector(states[i])
        for j in range(n_states):
            vector = states[j]
            array[i, j] = calculate_operator_inner_product(
                dual_vector, operator, vector
            )

    basis = (FundamentalPositionAxis(np.array([0]), n_states),)
    return {"array": array, "basis": basis, "dual_basis": basis}


def _localize_operator(
    wavepacket: WavepacketWithEigenvalues[_B0Inv, _B1Inv],
    operator: SingleBasisOperator[_B2Inv],
) -> list[WavepacketWithEigenvalues[_B0Inv, _B1Inv]]:
    states = [
        convert_state_vector_to_basis(state, operator["basis"])
        for state in get_all_eigenstates(wavepacket)
    ]
    operator_between_states = _get_operator_between_states(states, operator)
    eigenstates = calculate_eigenvectors_hermitian(operator_between_states)
    return [
        {
            "list_basis": wavepacket["list_basis"],
            "basis": wavepacket["basis"],
            "eigenvalues": wavepacket["eigenvalues"],
            "vectors": wavepacket["vectors"] * vector[:, np.newaxis],
        }
        for vector in eigenstates["vectors"]
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
    basis = basis_as_fundamental_position_basis(
        get_unfurled_basis(wavepacket["list_basis"], wavepacket["basis"])
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
    basis = basis_as_fundamental_position_basis(
        get_unfurled_basis(wavepackets[0]["list_basis"], wavepackets[0]["basis"])
    )
    states = [
        convert_state_vector_to_basis(state, basis)
        for wavepacket in wavepackets
        for state in get_all_eigenstates(wavepacket)
    ]
    operator_position = _get_position_operator(basis)
    operator = _get_operator_between_states(states, operator_position)
    eigenstates = calculate_eigenvectors_hermitian(operator)
    state_vectors = np.array([s["vector"] for s in states])
    return [
        {
            "basis": basis,
            "vector": np.tensordot(vector, state_vectors, axes=(0, 0)),
        }
        for vector in eigenstates["vectors"]
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
    list_shape = BasisUtil(wavepackets[0]["list_basis"]).shape
    states = [
        unfurl_wavepacket(
            localize_position_operator(wavepacket)[np.prod(list_shape) // 4]
        )
        for wavepacket in wavepackets
    ]
    operator_position = _get_position_operator(states[0]["basis"])
    operator = _get_operator_between_states(states, operator_position)  # type: ignore[arg-type]
    eigenstates = calculate_eigenvectors_hermitian(operator)
    state_vectors = np.array([s["vector"] for s in states])
    return [
        {
            "basis": states[0]["basis"],
            "vector": np.tensordot(vector, state_vectors, axes=(0, 0)),
        }
        for vector in eigenstates["vectors"]
    ]
