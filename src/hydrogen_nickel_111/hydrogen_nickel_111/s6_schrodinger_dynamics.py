from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from surface_potential_analysis.dynamics.incoherent_propagation.tunnelling_matrix import (
    resample_tunnelling_a_matrix,
)
from surface_potential_analysis.dynamics.stochastic_shrodinger_equation import (
    get_simplified_collapse_operators_from_a_matrix,
    solve_stochastic_schrodinger_equation,
)

from hydrogen_nickel_111.s6_a_calculation import get_tunnelling_a_matrix_hydrogen

if TYPE_CHECKING:
    from surface_potential_analysis.operator.operator import SingleBasisOperator
    from surface_potential_analysis.state_vector.state_vector import StateVector


def get_equilibrium_state_on_surface_hydrogen() -> None:
    a_matrix = get_tunnelling_a_matrix_hydrogen((5, 5), 6, 150)
    np.fill_diagonal(a_matrix["array"], 0)
    resampled = resample_tunnelling_a_matrix(a_matrix, (3, 3))

    collapse_operators = get_simplified_collapse_operators_from_a_matrix(resampled)

    hamiltonian: SingleBasisOperator[Any] = {
        "basis": resampled["basis"],
        "dual_basis": resampled["basis"],
        "array": np.zeros_like(resampled["array"]),
    }
    initial_state: StateVector[Any] = {
        "basis": resampled["basis"],
        "vector": np.zeros(hamiltonian["array"].shape[0]),
    }
    initial_state["vector"][0] = 1
    solve_stochastic_schrodinger_equation(
        initial_state, hamiltonian, collapse_operators
    )
