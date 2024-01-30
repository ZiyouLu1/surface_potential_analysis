from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import Boltzmann
from surface_potential_analysis.basis.basis import FundamentalBasis
from surface_potential_analysis.basis.stacked_basis import StackedBasis
from surface_potential_analysis.basis.time_basis_like import FundamentalTimeBasis
from surface_potential_analysis.dynamics.incoherent_propagation.eigenstates import (
    calculate_tunnelling_simulation_state,
)
from surface_potential_analysis.dynamics.incoherent_propagation.tunnelling_matrix import (
    density_matrix_list_as_probabilities,
    get_initial_pure_density_matrix_for_basis,
    get_tunnelling_m_matrix,
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
from surface_potential_analysis.operator.operator import (
    average_eigenvalues_list,
)
from surface_potential_analysis.operator.operator_list import as_flat_operator
from surface_potential_analysis.probability_vector.plot import (
    plot_total_probability_against_time,
)
from surface_potential_analysis.probability_vector.probability_vector import (
    from_state_vector_list,
    get_probability_along_axis,
)
from surface_potential_analysis.util.interpolation import pad_ft_points

from hydrogen_nickel_111.s6_a_calculation import get_tunnelling_a_matrix_hydrogen

from .s4_wavepacket import (
    get_wannier90_localized_split_bands_hamiltonian_hydrogen,
    get_wavepacket_hamiltonian_hydrogen,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from surface_potential_analysis.basis.basis_like import BasisLike
    from surface_potential_analysis.dynamics.tunnelling_basis import (
        TunnellingSimulationBasis,
    )
    from surface_potential_analysis.operator.operator import (
        SingleBasisDiagonalOperator,
        SingleBasisOperator,
    )
    from surface_potential_analysis.operator.operator_list import OperatorList
    from surface_potential_analysis.state_vector.state_vector import StateVector
    from surface_potential_analysis.wavepacket.wavepacket import (
        WavepacketWithEigenvalues,
    )

    _B0 = TypeVar("_B0", bound=BasisLike[Any, Any])
    _B0Inv = TypeVar("_B0Inv", bound=TunnellingSimulationBasis[Any, Any, Any])


def build_hamiltonian_from_wavepackets(
    wavepackets: list[WavepacketWithEigenvalues[Any, Any]],
    basis: _B0Inv,
) -> SingleBasisOperator[_B0Inv]:
    (n_x1, n_x2, _) = basis.shape
    array = np.zeros((*basis.shape, *basis.shape), np.complex128)
    for i, wavepacket in enumerate(wavepackets):
        sample_shape = wavepacket["basis"][0].shape
        h = pad_ft_points(
            np.fft.fftn(
                wavepacket["eigenvalue"].reshape(sample_shape),
                norm="ortho",
            ),
            (3, 3, 1),
            (0, 1, 2),
        )
        for hop, hop_val in enumerate(h.ravel()):
            array[:, :, i, :, :, i] += hop_val * build_hop_operator(hop, (n_x1, n_x2))
    return {
        "data": array.reshape(-1),
        "basis": StackedBasis(basis, basis),
    }


def get_hop_hamiltonian() -> (
    OperatorList[
        StackedBasis[
            FundamentalBasis[Literal[3]],
            FundamentalBasis[Literal[3]],
            FundamentalBasis[Literal[1]],
        ],
        FundamentalBasis[int],
        FundamentalBasis[int],
    ]
):
    hamiltonian = get_wannier90_localized_split_bands_hamiltonian_hydrogen()
    stacked = hamiltonian["data"].reshape(*hamiltonian["basis"][0].shape, -1)
    # Convert the list basis into 'position basis'
    transformed = np.fft.ifftn(stacked, norm="ortho")
    # TODO: check that coherent propagation in next neighbor unit cell is small
    # ie is this truncation valid??
    truncated = pad_ft_points(transformed, (3, 3, 1), (0, 1, 2))
    return {
        "basis": StackedBasis(
            StackedBasis(
                FundamentalBasis[Literal[3]](3),
                FundamentalBasis[Literal[3]](3),
                FundamentalBasis[Literal[1]](1),
            ),
            hamiltonian["basis"][1],
        ),
        "data": truncated.reshape(-1),
    }


def get_coherent_hamiltonian(
    basis: _B0Inv,
) -> SingleBasisOperator[_B0Inv]:
    hop_hamiltonian = get_hop_hamiltonian()
    (n_x1, n_x2, n_bands) = basis.shape
    assert hop_hamiltonian["basis"][1][0].n == n_bands

    hop_hamiltonian_stacked = hop_hamiltonian["data"].reshape(9, n_bands, n_bands)
    data = np.zeros((*basis.shape, *basis.shape), np.complex128)
    for n_0 in range(n_bands):
        for n_1 in range(n_bands):
            for hop in range(9):
                hop_val = hop_hamiltonian_stacked[hop, n_0, n_1]
                data[:, :, n_0, :, :, n_1] += hop_val * build_hop_operator(
                    hop, (n_x1, n_x2)
                )

    return {"basis": StackedBasis(basis, basis), "data": data.reshape(-1)}


def plot_expected_occupation_per_band(
    temperature: float,
    eigenvalues: SingleBasisDiagonalOperator[_B0],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    energy = eigenvalues["data"]
    factors = np.exp(-energy / (temperature * Boltzmann))
    factors /= np.sum(factors)
    for factor in factors:
        ax.axhline(y=factor)  # cSpell:disable-line
    return (fig, ax)


def plot_expected_occupation_per_band_hydrogen(
    temperature: float,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    eigenvalues = get_wavepacket_hamiltonian_hydrogen(8)

    band_average = as_flat_operator(average_eigenvalues_list(eigenvalues, axis=(0,)))
    return plot_expected_occupation_per_band(temperature, band_average, ax=ax)


def plot_occupation_on_surface_hydrogen() -> None:
    a_matrix = get_tunnelling_a_matrix_hydrogen((25, 25), 8, 150)

    collapse_operators = get_simplified_collapse_operators_from_a_matrix(a_matrix)

    hamiltonian = get_coherent_hamiltonian(a_matrix["basis"][0])

    initial_state: StateVector[Any] = {
        "basis": a_matrix["basis"][0],
        "data": np.zeros((hamiltonian["basis"][0].n,), dtype=np.complex128),
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
    a_matrix = get_tunnelling_a_matrix_hydrogen((3, 3), 8, 150)

    collapse_operators = get_simplified_collapse_operators_from_a_matrix(a_matrix)

    hamiltonian = get_coherent_hamiltonian(a_matrix["basis"][0])
    hamiltonian["data"] = np.zeros_like(hamiltonian["data"])

    initial_state: StateVector[Any] = {
        "basis": a_matrix["basis"][0],
        "data": np.zeros(hamiltonian["basis"][0].n, dtype=np.complex128),
    }
    initial_state["data"][0] = 1
    times = FundamentalTimeBasis(20000, 5e-10)
    states = solve_stochastic_schrodinger_equation(
        initial_state, times, hamiltonian, collapse_operators, n_trajectories=20
    )
    probabilities = from_state_vector_list(states)
    fig, ax, _ = plot_average_probability_per_band(probabilities)
    plot_expected_occupation_per_band_hydrogen(150, ax=ax)

    m_matrix = get_tunnelling_m_matrix(a_matrix)
    initial_state_incoherent = get_initial_pure_density_matrix_for_basis(
        m_matrix["basis"][0]
    )
    state = calculate_tunnelling_simulation_state(
        m_matrix, initial_state_incoherent, times.times
    )
    probability = density_matrix_list_as_probabilities(state)
    plot_probability_per_band(probability, ax=ax)

    fig.show()
    input()
