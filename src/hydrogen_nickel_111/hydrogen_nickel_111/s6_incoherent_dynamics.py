from __future__ import annotations

import numpy as np
from surface_potential_analysis.dynamics.incoherent_propagation.eigenstates import (
    calculate_equilibrium_state,
    calculate_tunnelling_simulation_state,
)
from surface_potential_analysis.dynamics.incoherent_propagation.tunnelling_matrix import (
    density_matrix_list_as_probabilities,
    get_initial_pure_density_matrix_for_basis,
    get_tunnelling_m_matrix,
)
from surface_potential_analysis.dynamics.plot import (
    plot_probability_per_band,
    plot_probability_per_site,
)
from surface_potential_analysis.operator.operator import sum_diagonal_operator_over_axes

from .s6_a_calculation import (
    get_tunnelling_a_matrix_deuterium,
    get_tunnelling_a_matrix_hydrogen,
)


def test_normalization_of_m_matrix_hydrogen() -> None:
    rng = np.random.default_rng()
    a_matrix = get_tunnelling_a_matrix_hydrogen((5, 5), 2, 150)
    m_matrix = get_tunnelling_m_matrix(a_matrix)

    initial = rng.random(m_matrix["array"].shape[0])
    initial /= np.linalg.norm(initial)
    actual = np.sum(np.tensordot(m_matrix["array"], initial, (1, 0)))  # type: ignore[var-annotated]
    np.testing.assert_array_almost_equal(0, actual)

    np.testing.assert_array_almost_equal(0, np.sum(m_matrix["array"], axis=0))

    np.testing.assert_array_equal(1, np.diag(m_matrix["array"]) <= 0)
    np.testing.assert_array_equal(1, a_matrix["array"] >= 0)


def test_reduced_band_matrix_hydrogen() -> None:
    a_matrix = get_tunnelling_a_matrix_hydrogen((5, 5), 2, 150)
    expected = get_tunnelling_m_matrix(a_matrix)

    a_matrix_6 = get_tunnelling_a_matrix_hydrogen((25, 25), 6, 150)
    actual = get_tunnelling_m_matrix(a_matrix_6, 2)
    np.testing.assert_array_almost_equal(actual["array"], expected["array"])


def get_equilibrium_state_on_surface_hydrogen() -> None:
    a_matrix = get_tunnelling_a_matrix_hydrogen((5, 5), 6, 150)
    m_matrix = get_tunnelling_m_matrix(a_matrix)
    state = calculate_equilibrium_state(m_matrix)
    print(state["vector"])  # noqa: T201
    print(sum_diagonal_operator_over_axes(state, (0, 1))["vector"])  # noqa: T201

    m_matrix = get_tunnelling_m_matrix(a_matrix, 2)
    state = calculate_equilibrium_state(m_matrix)
    print(state["vector"])  # noqa: T201
    print(sum_diagonal_operator_over_axes(state, (0, 1))["vector"])  # noqa: T201


def plot_occupation_on_surface_hydrogen() -> None:
    a_matrix = get_tunnelling_a_matrix_hydrogen((25, 25), 6, 150)
    m_matrix = get_tunnelling_m_matrix(a_matrix)
    initial_state = get_initial_pure_density_matrix_for_basis(m_matrix["basis"])
    times = np.linspace(0, 9e-10, 1000)
    state = calculate_tunnelling_simulation_state(m_matrix, initial_state, times)
    probabilities = density_matrix_list_as_probabilities(state)

    fig, _, _ = plot_probability_per_band(probabilities)
    fig.show()

    fig, _, _ = plot_probability_per_site(probabilities)
    fig.show()

    m_matrix_2_band = get_tunnelling_m_matrix(a_matrix, 2)
    initial_state_2_band = get_initial_pure_density_matrix_for_basis(
        m_matrix_2_band["basis"]
    )
    times = np.linspace(0, 9e-10, 1000)
    state = calculate_tunnelling_simulation_state(
        m_matrix_2_band, initial_state_2_band, times
    )
    probabilities = density_matrix_list_as_probabilities(state)

    fig, _, _ = plot_probability_per_band(probabilities)
    fig.show()

    fig, _, _ = plot_probability_per_site(probabilities)
    fig.show()
    input()


def plot_occupation_on_surface_deuterium() -> None:
    a_matrix = get_tunnelling_a_matrix_deuterium((25, 25), 6, 150)
    m_matrix = get_tunnelling_m_matrix(a_matrix)
    initial_state = get_initial_pure_density_matrix_for_basis(m_matrix["basis"])
    times = np.linspace(0, 9e-10, 1000)
    state = calculate_tunnelling_simulation_state(m_matrix, initial_state, times)
    probabilities = density_matrix_list_as_probabilities(state)

    fig, _, _ = plot_probability_per_band(probabilities)
    fig.show()

    fig, _, _ = plot_probability_per_site(probabilities)
    fig.show()

    m_matrix_2_band = get_tunnelling_m_matrix(a_matrix, 2)
    initial_state_2_band = get_initial_pure_density_matrix_for_basis(
        m_matrix_2_band["basis"]
    )
    times = np.linspace(0, 9e-10, 1000)
    state = calculate_tunnelling_simulation_state(
        m_matrix_2_band, initial_state_2_band, times
    )
    probabilities = density_matrix_list_as_probabilities(state)

    fig, _, _ = plot_probability_per_band(probabilities)
    fig.show()

    fig, _, _ = plot_probability_per_site(probabilities)
    fig.show()
    input()
