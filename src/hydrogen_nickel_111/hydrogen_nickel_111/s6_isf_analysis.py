from __future__ import annotations

from typing import Literal, TypeVar

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.scale import FuncScale
from surface_potential_analysis.basis.util import AxisWithLengthBasisUtil
from surface_potential_analysis.dynamics.incoherent_propagation.isf import (
    calculate_equilibrium_state_averaged_isf,
    calculate_isf_at_times,
    fit_isf_to_double_exponential,
)
from surface_potential_analysis.dynamics.incoherent_propagation.isf_plot import (
    plot_isf_against_time,
    plot_isf_fit_against_time,
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
        m_matrix = get_tunnelling_m_matrix(a_matrix)
        isf = calculate_equilibrium_state_averaged_isf(m_matrix, times, dk)
        fit = fit_isf_to_double_exponential(isf, times)
        rates[0, i] = fit.fast_rate
        rates[1, i] = fit.slow_rate
    return rates  # type: ignore[no-any-return]


def plot_tunnelling_rate_hydrogen() -> None:
    temperatures = np.array([125, 150, 175, 200, 225])

    fig, ax = plt.subplots()

    fast_rates, slow_rates = calculate_rates_hydrogen(temperatures)
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
    times = np.linspace(0, 90e-10, 1000)
    dk = get_jianding_isf_dk()

    fig, ax = plt.subplots()

    m_matrix = get_tunnelling_m_matrix(a_matrix)
    initial_state = get_initial_pure_density_matrix_for_basis(m_matrix["basis"])
    isf = calculate_isf_at_times(m_matrix, initial_state, times, dk)
    fig, _, _ = plot_isf_against_time(isf, times, ax=ax)

    isf_fit = fit_isf_to_double_exponential(isf, times)
    fig, _, _ = plot_isf_fit_against_time(isf_fit, times, ax=ax)

    m_matrix = get_m_matrix_reduced_bands(m_matrix, 2)
    initial_state = get_initial_pure_density_matrix_for_basis(m_matrix["basis"])
    isf = calculate_isf_at_times(m_matrix, initial_state, times, dk)
    fig, _, _ = plot_isf_against_time(isf, times, ax=ax)
    fig.show()
    input()


def compare_isf_initial_condition() -> None:
    a_matrix = get_tunnelling_a_matrix_hydrogen((5, 5), 6, 150)
    m_matrix = get_tunnelling_m_matrix(a_matrix)

    fig, ax = plt.subplots()
    dk = get_jianding_isf_dk()

    initial_state = get_initial_pure_density_matrix_for_basis(
        m_matrix["basis"], (0, 0, 0)
    )
    times = np.linspace(0, 90e-10, 1000)
    isf = calculate_isf_at_times(m_matrix, initial_state, times, dk)
    _, _, line = plot_isf_against_time(isf, times, ax=ax)
    line.set_label("Initially FCC")

    initial_state = get_initial_pure_density_matrix_for_basis(
        m_matrix["basis"], (0, 0, 1)
    )
    isf = calculate_isf_at_times(m_matrix, initial_state, times, dk)
    _, _, line = plot_isf_against_time(isf, times, ax=ax)
    line.set_label("Initially HCP")

    ax.legend()
    fig.show()
    input()
