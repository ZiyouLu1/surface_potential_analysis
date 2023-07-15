from __future__ import annotations

from typing import Literal, TypeVar

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.scale import FuncScale
from surface_potential_analysis.basis.util import AxisWithLengthBasisUtil
from surface_potential_analysis.dynamics.incoherent_propagation.eigenstates import (
    calculate_tunnelling_simulation_state,
)
from surface_potential_analysis.dynamics.incoherent_propagation.isf import (
    calculate_isf_approximate_locations,
    fit_isf_to_double_exponential,
    get_isf_from_fit,
)
from surface_potential_analysis.dynamics.incoherent_propagation.isf_plot import (
    plot_isf_against_time,
)
from surface_potential_analysis.dynamics.incoherent_propagation.tunnelling_basis import (
    TunnellingSimulationBandsAxis,
)
from surface_potential_analysis.dynamics.incoherent_propagation.tunnelling_matrix import (
    get_initial_pure_density_matrix_for_basis,
    get_m_matrix_reduced_bands,
    get_tunnelling_m_matrix,
)

from hydrogen_nickel_111.s6_a_calculation import get_tunnelling_a_matrix_hydrogen

from .s4_wavepacket import get_wavepacket_hydrogen

_L0Inv = TypeVar("_L0Inv", bound=int)


def get_jianding_isf_dk() -> np.ndarray[tuple[Literal[2]], np.dtype[np.float_]]:
    basis = get_wavepacket_hydrogen(0)["basis"]

    util = AxisWithLengthBasisUtil(basis)
    dk = util.delta_x[0] + util.delta_x[1]
    dk /= np.linalg.norm(dk)
    dk *= 0.8 * 10**10
    return dk[:2]  # type: ignore[no-any-return]


def calculate_rates_hydrogen(
    temperatures: np.ndarray[tuple[_L0Inv], np.dtype[np.float_]]
) -> np.ndarray[tuple[Literal[2], _L0Inv], np.dtype[np.float_]]:
    times = np.linspace(0, 9e-9, 1000)
    dk = get_jianding_isf_dk()

    rates = np.zeros((2, temperatures.shape[0]))
    for i, t in enumerate(temperatures):
        a_matrix = get_tunnelling_a_matrix_hydrogen((5, 5), 6, t)
        a_matrix["basis"] = (
            a_matrix["basis"][0],
            a_matrix["basis"][1],
            TunnellingSimulationBandsAxis(
                a_matrix["basis"][2].locations,
                tuple(x[0:2] for x in a_matrix["basis"][2].unit_cell),
            ),
        )
        m_matrix = get_tunnelling_m_matrix(a_matrix)
        initial_state = get_initial_pure_density_matrix_for_basis(
            m_matrix["basis"], (0, 0, 0)
        )
        state = calculate_tunnelling_simulation_state(m_matrix, initial_state, times)
        isf = calculate_isf_approximate_locations(initial_state, state, dk)
        fit = fit_isf_to_double_exponential(isf, times)
        rates[0, i] = fit.fast_rate
        rates[1, i] = fit.slow_rate
    return rates  # type: ignore[no-any-return]


def plot_tunnelling_rate_hydrogen() -> None:
    temperatures = np.array([125, 150, 175, 200, 225])
    fast_rates, slow_rates = calculate_rates_hydrogen(temperatures)
    fig, ax = plt.subplots()

    (line,) = ax.plot(temperatures, fast_rates)
    line.set_label("Fast Rate")

    (line,) = ax.plot(temperatures, slow_rates)
    line.set_label("Slow Rate")

    ax.legend()
    fig.show()
    scale = FuncScale(ax, [lambda x: 1 / x, lambda x: 1 / x])
    ax.set_xscale(scale)
    ax.set_yscale("log")
    input()


def plot_isf_hydrogen() -> None:
    a_matrix = get_tunnelling_a_matrix_hydrogen((5, 5), 6, 150)
    a_matrix["basis"] = (
        a_matrix["basis"][0],
        a_matrix["basis"][1],
        TunnellingSimulationBandsAxis(
            a_matrix["basis"][2].locations,
            tuple(x[0:2] for x in a_matrix["basis"][2].unit_cell),
        ),
    )
    m_matrix = get_tunnelling_m_matrix(a_matrix)
    initial_state = get_initial_pure_density_matrix_for_basis(m_matrix["basis"])
    times = np.linspace(0, 90e-10, 1000)
    state = calculate_tunnelling_simulation_state(m_matrix, initial_state, times)

    m_matrix_2_band = get_m_matrix_reduced_bands(m_matrix, 2)
    initial_state_2_band = get_initial_pure_density_matrix_for_basis(
        m_matrix_2_band["basis"]
    )
    state_2_band = calculate_tunnelling_simulation_state(
        m_matrix_2_band, initial_state_2_band, times
    )

    fig, ax = plt.subplots()
    dk = get_jianding_isf_dk()
    isf = calculate_isf_approximate_locations(initial_state, state, dk)
    fig, _, _ = plot_isf_against_time(isf, times, ax=ax)
    isf_fit = get_isf_from_fit(fit_isf_to_double_exponential(isf, times), times)
    fig, _, _ = plot_isf_against_time(isf_fit, times, ax=ax)
    isf = calculate_isf_approximate_locations(initial_state_2_band, state_2_band, dk)
    fig, _, _ = plot_isf_against_time(isf, times, ax=ax)
    fig.show()
    input()


def compare_isf_initial_condition() -> None:
    a_matrix = get_tunnelling_a_matrix_hydrogen((5, 5), 6, 150)
    a_matrix["basis"] = (
        a_matrix["basis"][0],
        a_matrix["basis"][1],
        TunnellingSimulationBandsAxis(
            a_matrix["basis"][2].locations,
            tuple(x[0:2] for x in a_matrix["basis"][2].unit_cell),
        ),
    )
    m_matrix = get_tunnelling_m_matrix(a_matrix)

    fig, ax = plt.subplots()
    dk = get_jianding_isf_dk()

    initial_state = get_initial_pure_density_matrix_for_basis(
        m_matrix["basis"], (0, 0, 0)
    )
    times = np.linspace(0, 90e-10, 1000)
    state = calculate_tunnelling_simulation_state(m_matrix, initial_state, times)
    isf = calculate_isf_approximate_locations(initial_state, state, dk)
    _, _, line = plot_isf_against_time(isf, times, ax=ax)
    line.set_label("Initially FCC")

    initial_state = get_initial_pure_density_matrix_for_basis(
        m_matrix["basis"], (0, 0, 1)
    )
    times = np.linspace(0, 90e-10, 1000)
    state = calculate_tunnelling_simulation_state(m_matrix, initial_state, times)
    isf = calculate_isf_approximate_locations(initial_state, state, dk)
    _, _, line = plot_isf_against_time(isf, times, ax=ax)
    line.set_label("Initially HCP")

    ax.legend()
    fig.show()
    input()
