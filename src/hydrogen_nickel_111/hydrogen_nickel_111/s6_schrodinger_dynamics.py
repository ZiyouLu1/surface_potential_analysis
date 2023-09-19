from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import Boltzmann
from surface_potential_analysis.axis.time_axis_like import FundamentalTimeBasis
from surface_potential_analysis.dynamics.incoherent_propagation.eigenstates import (
    calculate_tunnelling_simulation_state,
)
from surface_potential_analysis.dynamics.incoherent_propagation.tunnelling_matrix import (
    density_matrix_list_as_probabilities,
    get_initial_pure_density_matrix_for_basis,
    get_tunnelling_m_matrix,
    resample_tunnelling_a_matrix,
)
from surface_potential_analysis.dynamics.plot import (
    plot_average_probability_per_band,
    plot_probability_per_band,
    plot_probability_per_site,
)
from surface_potential_analysis.dynamics.stochastic_schrodinger.solve import (
    get_simplified_collapse_operators_from_a_matrix,
    solve_stochastic_schrodinger_equation,
)
from surface_potential_analysis.dynamics.util import build_hop_operator
from surface_potential_analysis.probability_vector.plot import (
    plot_total_probability_against_time,
)
from surface_potential_analysis.probability_vector.probability_vector import (
    from_state_vector_list,
    get_probability_along_axis,
)
from surface_potential_analysis.util.interpolation import pad_ft_points

from hydrogen_nickel_111.s6_a_calculation import get_tunnelling_a_matrix_hydrogen

from .s4_wavepacket import get_all_wavepackets_hydrogen

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from surface_potential_analysis.dynamics.tunnelling_basis import (
        TunnellingSimulationBasis,
    )
    from surface_potential_analysis.operator.operator import SingleBasisOperator
    from surface_potential_analysis.state_vector.state_vector import StateVector
    from surface_potential_analysis.wavepacket.wavepacket import (
        WavepacketWithEigenvalues,
    )

    _B0Inv = TypeVar("_B0Inv", bound=TunnellingSimulationBasis[Any, Any, Any])


def build_hamiltonian_from_wavepackets(
    wavepackets: list[WavepacketWithEigenvalues[Any, Any]],
    basis: _B0Inv,
) -> SingleBasisOperator[_B0Inv]:
    (n_x1, n_x2, _) = basis.shape
    array = np.zeros((*basis.shape, *basis.shape), np.complex_)
    for i, wavepacket in enumerate(wavepackets):
        sample_shape = wavepacket["basis"][0].shape
        h = pad_ft_points(
            np.fft.fftn(
                wavepacket["eigenvalues"].reshape(sample_shape),
                norm="ortho",
            ),
            (3, 3, 1),
            (0, 1, 2),
        )
        for hop, hop_val in enumerate(h.ravel()):
            array[:, :, i, :, :, i] += hop_val * build_hop_operator(hop, (n_x1, n_x2))
    return {
        "array": array.reshape(-1),
        "basis": StackedBasis(basis, basis),
    }


def build_hamiltonian_hydrogen(
    basis: _B0Inv,
) -> SingleBasisOperator[_B0Inv]:
    return build_hamiltonian_from_wavepackets(
        get_all_wavepackets_hydrogen()[: basis[2].fundamental_n], basis
    )


def plot_expected_occupation_per_band(
    temperature: float,
    wavepackets: list[WavepacketWithEigenvalues[Any, Any]],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    average_energy = np.average(
        [wavepacket["eigenvalues"] for wavepacket in wavepackets], axis=1
    )
    factors = np.exp(-average_energy / (temperature * Boltzmann))
    factors /= np.sum(factors)
    for factor in factors:
        ax.axhline(y=factor)  # cSpell:disable-line
    return (fig, ax)


def plot_expected_occupation_per_band_hydrogen(
    temperature: float,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    wavepackets = get_all_wavepackets_hydrogen()
    return plot_expected_occupation_per_band(temperature, wavepackets, ax=ax)


def plot_occupation_on_surface_hydrogen() -> None:
    a_matrix = get_tunnelling_a_matrix_hydrogen((25, 25), 6, 150)
    np.fill_diagonal(a_matrix["array"], 0)
    resampled = resample_tunnelling_a_matrix(a_matrix, (3, 3), 6)

    collapse_operators = get_simplified_collapse_operators_from_a_matrix(resampled)

    hamiltonian = build_hamiltonian_hydrogen(resampled["basis"])

    initial_state: StateVector[Any] = {
        "basis": resampled["basis"],
        "data": np.zeros(hamiltonian["array"].shape[0]),
    }
    initial_state["data"][0] = 1
    times = FundamentalTimeBasis(20000, 5e-10)
    states = solve_stochastic_schrodinger_equation(
        initial_state, times, hamiltonian, collapse_operators, n_trajectories=20
    )
    probabilities = from_state_vector_list(states)

    fig, ax, _ = plot_average_probability_per_band(probabilities)
    plot_expected_occupation_per_band_hydrogen(150, ax=ax)
    fig.show()

    probability_0 = get_probability_along_axis(probabilities, (1,), (0,))

    fig, ax, _ = plot_probability_per_band(probability_0)
    plot_expected_occupation_per_band_hydrogen(150, ax=ax)
    fig.show()

    fig, _, _ = plot_probability_per_site(probability_0)
    fig.show()

    fig, _, _ = plot_total_probability_against_time(probability_0)
    fig.show()
    input()


def plot_incoherent_occupation_comparison_hydrogen() -> None:
    a_matrix = get_tunnelling_a_matrix_hydrogen((25, 25), 6, 150)
    np.fill_diagonal(a_matrix["array"], 0)
    resampled = resample_tunnelling_a_matrix(a_matrix, (3, 3), 6)

    collapse_operators = get_simplified_collapse_operators_from_a_matrix(resampled)

    hamiltonian = build_hamiltonian_hydrogen(resampled["basis"])
    hamiltonian["array"] = np.zeros_like(hamiltonian["array"])

    initial_state: StateVector[Any] = {
        "basis": resampled["basis"],
        "data": np.zeros(hamiltonian["array"].shape[0]),
    }
    initial_state["data"][0] = 1
    times = FundamentalTimeBasis(20000, 5e-10)
    states = solve_stochastic_schrodinger_equation(
        initial_state, times, hamiltonian, collapse_operators, n_trajectories=20
    )
    probabilities = from_state_vector_list(states)
    fig, ax, _ = plot_average_probability_per_band(probabilities)
    plot_expected_occupation_per_band_hydrogen(150, ax=ax)

    m_matrix = get_tunnelling_m_matrix(resampled)
    initial_state_incoherent = get_initial_pure_density_matrix_for_basis(
        m_matrix["basis"]
    )
    state = calculate_tunnelling_simulation_state(
        m_matrix, initial_state_incoherent, times.times
    )
    probability = density_matrix_list_as_probabilities(state)
    plot_probability_per_band(probability, ax=ax)

    fig.show()
    input()
