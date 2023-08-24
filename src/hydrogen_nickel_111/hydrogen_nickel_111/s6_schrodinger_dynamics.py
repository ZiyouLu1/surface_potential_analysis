from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from surface_potential_analysis.dynamics.incoherent_propagation.tunnelling_matrix import (
    downsample_tunnelling_a_matrix,
)
from surface_potential_analysis.dynamics.stochastic_shrodinger_equation import (
    get_collapse_operators_from_a_matrix,
    solve_stochastic_schrodinger_equation,
)

from hydrogen_nickel_111.s6_a_calculation import get_tunnelling_a_matrix_hydrogen

if TYPE_CHECKING:
    from surface_potential_analysis.operator.operator import SingleBasisOperator
    from surface_potential_analysis.state_vector.state_vector import StateVector


def get_equilibrium_state_on_surface_hydrogen() -> None:
    a_matrix = get_tunnelling_a_matrix_hydrogen((5, 5), 6, 150)
    downsampled = downsample_tunnelling_a_matrix(a_matrix, (2, 2))
    collapse_operators = get_collapse_operators_from_a_matrix(downsampled)
    hamiltonian: SingleBasisOperator = {
        "basis": downsampled["basis"],
        "dual_basis": downsampled["basis"],
        "array": np.zeros_like(downsampled["array"]),
    }
    initial_state: StateVector = {
        "basis": downsampled["basis"],
        "vector": np.zeros(hamiltonian["array"].shape[0]),
    }
    initial_state["vector"][0] = 1
    state = solve_stochastic_schrodinger_equation(
        initial_state, hamiltonian, collapse_operators
    )
    print(state["vector"])  # noqa: T201
    print(sum_diagonal_operator_over_axes(state, (0, 1))["vector"])  # noqa: T201

    m_matrix = get_tunnelling_m_matrix(a_matrix, 2)
    state = calculate_equilibrium_state(m_matrix)
    print(state["vector"])  # noqa: T201
    print(sum_diagonal_operator_over_axes(state, (0, 1))["vector"])  # noqa: T201
