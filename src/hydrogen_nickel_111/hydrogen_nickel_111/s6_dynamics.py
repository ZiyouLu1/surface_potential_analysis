from __future__ import annotations

from typing import Literal

import numpy as np
from surface_potential_analysis.dynamics.incoherent_propagation.eigenstates import (
    calculate_equilibrium_state,
    calculate_tunnelling_simulation_state,
)
from surface_potential_analysis.dynamics.incoherent_propagation.plot import (
    plot_occupation_per_band,
    plot_occupation_per_site,
)
from surface_potential_analysis.dynamics.incoherent_propagation.tunnelling_basis import (
    TunnellingSimulationBandsAxis,
)
from surface_potential_analysis.dynamics.incoherent_propagation.tunnelling_matrix import (
    get_initial_pure_density_matrix_for_basis,
    get_m_matrix_reduced_bands,
    get_tunnelling_m_matrix,
)

from hydrogen_nickel_111.s6_a_calculation import (
    get_tunnelling_a_matrix_deuterium,
    get_tunnelling_a_matrix_hydrogen,
)

from .s4_wavepacket import (
    get_all_wavepackets_deuterium,
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


def get_equilibrium_state_on_surface_hydrogen() -> None:
    a_matrix = get_tunnelling_a_matrix_hydrogen((5, 5), 2, 150)
    m_matrix = get_tunnelling_m_matrix(a_matrix)
    state = calculate_equilibrium_state(m_matrix)
    print(state["vector"])  # noqa: T201
    print(np.sum(state["vector"]))  # noqa: T201


def plot_occupation_on_surface_hydrogen() -> None:
    a_matrix = get_tunnelling_a_matrix_hydrogen((5, 5), 6, 150)
    m_matrix = get_tunnelling_m_matrix(a_matrix)
    initial_state = get_initial_pure_density_matrix_for_basis(m_matrix["basis"])
    times = np.linspace(0, 9e-10, 1000)
    state = calculate_tunnelling_simulation_state(m_matrix, initial_state, times)

    fig, ax = plot_occupation_per_band(state, times)
    fig.show()

    fig, ax = plot_occupation_per_site(state, times)
    fig.show()

    m_matrix_2_band = get_m_matrix_reduced_bands(m_matrix, 2)
    initial_state_2_band = get_initial_pure_density_matrix_for_basis(
        m_matrix_2_band["basis"]
    )
    times = np.linspace(0, 9e-10, 1000)
    state = calculate_tunnelling_simulation_state(
        m_matrix_2_band, initial_state_2_band, times
    )

    fig, ax = plot_occupation_per_band(state, times)
    fig.show()

    fig, ax = plot_occupation_per_site(state, times)
    fig.show()
    input()


def get_simulated_state_on_surface_hydrogen() -> None:
    a_matrix = get_tunnelling_a_matrix_hydrogen((5, 5), 2, 150)
    m_matrix = get_tunnelling_m_matrix(a_matrix)
    initial_state = get_initial_pure_density_matrix_for_basis(m_matrix["basis"])
    state = calculate_tunnelling_simulation_state(
        m_matrix, initial_state, np.array([99999])
    )

    print(state["vectors"])  # noqa: T201
    print(np.sum(state["vectors"]))  # noqa: T201


def plot_occupation_on_surface_deuterium() -> None:
    a_matrix = get_tunnelling_a_matrix_deuterium((5, 5), 6, 150)
    bands_axis = TunnellingSimulationBandsAxis[Literal[6]].from_wavepackets(
        get_all_wavepackets_deuterium()[0:6]
    )
    a_matrix["basis"] = (a_matrix["basis"][0], a_matrix["basis"][1], bands_axis)
    m_matrix = get_tunnelling_m_matrix(a_matrix)
    initial_state = get_initial_pure_density_matrix_for_basis(m_matrix["basis"])
    times = np.linspace(0, 9e-10, 1000)
    state = calculate_tunnelling_simulation_state(m_matrix, initial_state, times)

    fig, ax = plot_occupation_per_band(state, times)
    fig.show()

    fig, ax = plot_occupation_per_site(state, times)
    fig.show()

    m_matrix_2_band = get_m_matrix_reduced_bands(m_matrix, 2)
    initial_state_2_band = get_initial_pure_density_matrix_for_basis(
        m_matrix_2_band["basis"]
    )
    times = np.linspace(0, 9e-10, 1000)
    state = calculate_tunnelling_simulation_state(
        m_matrix_2_band, initial_state_2_band, times
    )

    fig, ax = plot_occupation_per_band(state, times)
    fig.show()

    fig, ax = plot_occupation_per_site(state, times)
    fig.show()
    input()
