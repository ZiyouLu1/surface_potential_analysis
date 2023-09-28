from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
from surface_potential_analysis.basis.time_basis_like import (
    EvenlySpacedTimeBasis,
    FundamentalTimeBasis,
)
from surface_potential_analysis.dynamics.isf import calculate_isf_approximate_locations
from surface_potential_analysis.dynamics.isf_plot import plot_isf_against_time
from surface_potential_analysis.dynamics.stochastic_schrodinger.solve import (
    get_simplified_collapse_operators_from_a_matrix,
    solve_stochastic_schrodinger_equation,
)
from surface_potential_analysis.probability_vector.probability_vector import (
    average_probabilities,
    from_state_vector,
    from_state_vector_list,
    get_probability,
)
from surface_potential_analysis.util.decorators import npy_cached

from hydrogen_nickel_111.s6_a_calculation import get_tunnelling_a_matrix_hydrogen
from hydrogen_nickel_111.s6_incoherent_isf import get_jianding_isf_112bar
from hydrogen_nickel_111.s6_schrodinger_dynamics import build_hamiltonian_hydrogen

from .surface_data import get_data_path

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import FundamentalBasis
    from surface_potential_analysis.state_vector.state_vector import StateVector
    from surface_potential_analysis.state_vector.state_vector_list import (
        StateVectorList,
    )

    _AX0Inv = TypeVar("_AX0Inv", bound=EvenlySpacedTimeBasis[Any, Any, Any])


@npy_cached(
    lambda temperature, times: get_data_path(  # noqa: ARG005
        f"dynamics/see_simulation_{temperature}K.npy"
    ),
    load_pickle=True,
)
def get_simulation_at_temperature(
    temperature: float, times: _AX0Inv
) -> StateVectorList[tuple[FundamentalBasis[Literal[4]], _AX0Inv], Any]:
    a_matrix = get_tunnelling_a_matrix_hydrogen((12, 12), 6, temperature)
    np.fill_diagonal(a_matrix["array"], 0)
    collapse_operators = get_simplified_collapse_operators_from_a_matrix(a_matrix)

    hamiltonian = build_hamiltonian_hydrogen(a_matrix["basis"])
    initial_state: StateVector[Any] = {
        "basis": a_matrix["basis"],
        "data": np.zeros(hamiltonian["array"].shape[0]),
    }
    initial_state["data"][0] = 1
    return solve_stochastic_schrodinger_equation(
        initial_state, times, hamiltonian, collapse_operators, n_trajectories=4
    )


def plot_average_isf_against_time() -> None:
    a_matrix = get_tunnelling_a_matrix_hydrogen((6, 6), 6, 150)
    np.fill_diagonal(a_matrix["array"], 0)

    collapse_operators = get_simplified_collapse_operators_from_a_matrix(a_matrix)

    hamiltonian = build_hamiltonian_hydrogen(a_matrix["basis"])
    hamiltonian["array"] = np.zeros_like(hamiltonian["array"])

    initial_state: StateVector[Any] = {
        "basis": a_matrix["basis"],
        "data": np.zeros(hamiltonian["array"].shape[0]),
    }
    initial_state["data"][0] = 1
    times = FundamentalTimeBasis(20000, 8e-10)
    states = solve_stochastic_schrodinger_equation(
        initial_state, times, hamiltonian, collapse_operators, n_trajectories=20
    )
    probabilities = from_state_vector_list(states)

    isf = calculate_isf_approximate_locations(
        from_state_vector(initial_state),
        average_probabilities(probabilities, (0,)),
        get_jianding_isf_112bar(),
    )
    fig, _, _ = plot_isf_against_time(isf)
    fig.show()
    input()


def plot_average_isf_all_temperatures() -> None:
    temperatures = np.array([100, 125, 150, 175, 200, 225, 250])
    times = [
        EvenlySpacedTimeBasis[Any, Any](2000, 2000, 0, 6e-8),
        EvenlySpacedTimeBasis(2000, 200, 0, 20e-9),
        EvenlySpacedTimeBasis(2000, 200, 0, 6e-9),
        EvenlySpacedTimeBasis(2000, 20, 0, 22e-10),
        EvenlySpacedTimeBasis(2000, 10, 0, 12e-10),
        EvenlySpacedTimeBasis(2000, 10, 0, 6e-10),
        EvenlySpacedTimeBasis(2000, 10, 0, 6e-10),
    ]
    for temperature, time in list(zip(temperatures, times, strict=True))[::-1]:
        states = get_simulation_at_temperature(temperature, time)
        continue
        probabilities = from_state_vector_list(states)

        isf = calculate_isf_approximate_locations(
            get_probability(probabilities, 0),
            average_probabilities(probabilities, (0,)),
            get_jianding_isf_112bar(),
        )
        fig, _, _ = plot_isf_against_time(isf)
        fig.show()
        input()
