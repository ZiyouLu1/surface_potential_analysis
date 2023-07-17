from __future__ import annotations

from typing import Literal, TypeVar

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.scale import FuncScale
from scipy.constants import Boltzmann
from surface_potential_analysis.basis.util import AxisWithLengthBasisUtil
from surface_potential_analysis.dynamics.hermitian_gamma_integral import (
    calculate_hermitian_gamma_occupation_integral,
)
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
from surface_potential_analysis.util.constants import FERMI_WAVEVECTOR

from hydrogen_nickel_111.experimental_data import get_experiment_data
from hydrogen_nickel_111.s6_a_calculation import get_tunnelling_a_matrix_hydrogen

from .s4_wavepacket import get_hydrogen_energy_difference, get_wavepacket_hydrogen

_L0Inv = TypeVar("_L0Inv", bound=int)


def get_jianding_isf_dk() -> np.ndarray[tuple[Literal[2]], np.dtype[np.float_]]:
    basis = get_wavepacket_hydrogen(0)["basis"]

    util = AxisWithLengthBasisUtil(basis)
    dk = util.delta_x[0] + util.delta_x[1]
    dk /= np.linalg.norm(dk)
    dk *= 0.8 * 10**10
    return dk[:2]  # type: ignore[no-any-return]


def calculate_rates_hydrogen(
    temperatures: np.ndarray[tuple[_L0Inv], np.dtype[np.float_]], n_bands: int = 6
) -> np.ndarray[tuple[Literal[2], _L0Inv], np.dtype[np.float_]]:
    times = np.linspace(0, 9e-9, 1000)
    dk = get_jianding_isf_dk()

    rates = np.zeros((2, temperatures.shape[0]))
    for i, t in enumerate(temperatures):
        a_matrix = get_tunnelling_a_matrix_hydrogen((5, 5), 6, t)
        m_matrix = get_tunnelling_m_matrix(a_matrix)
        m_matrix = get_m_matrix_reduced_bands(m_matrix, n_bands)
        isf = calculate_equilibrium_state_averaged_isf(m_matrix, times, dk)
        fit = fit_isf_to_double_exponential(isf, times, measure="real")
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

    fast_rates, slow_rates = calculate_rates_hydrogen(temperatures, n_bands=2)
    (line,) = ax.plot(temperatures, fast_rates)
    line.set_label("Fast Rate 2 band")

    (line,) = ax.plot(temperatures, slow_rates)
    line.set_label("Slow Rate 2 band")

    data = get_experiment_data()
    ax.errorbar(
        data["temperature"],
        data["rate"],
        yerr=[data["rate"] - data["lower_error"], data["upper_error"] - data["rate"]],
    )

    scale = FuncScale(ax, [lambda x: 1 / x, lambda x: 1 / x])
    ax.set_xscale(scale)
    ax.set_yscale("log")
    ax.legend()
    fig.show()
    input()


def plot_fast_slow_rate_ratios() -> None:
    """Plot the ratio between the fast and slow rate, and compare to the boltzmann distribution."""
    temperatures = np.array([125, 150, 175, 200, 225])
    fig, ax = plt.subplots()

    fast_rates, slow_rates = calculate_rates_hydrogen(temperatures)
    (line,) = ax.plot(temperatures, np.log(fast_rates / slow_rates))
    line.set_linestyle("")
    line.set_marker("x")
    line.set_label("6 band")

    fast_rates, slow_rates = calculate_rates_hydrogen(temperatures, n_bands=2)
    (line,) = ax.plot(temperatures, np.log(fast_rates / slow_rates))
    line.set_linestyle("")
    line.set_marker("x")
    line.set_label("2 band")

    non_isf_ratio = [
        calculate_hermitian_gamma_occupation_integral(
            get_hydrogen_energy_difference(0, 1),
            FERMI_WAVEVECTOR["NICKEL"],
            Boltzmann * t,
        )
        / calculate_hermitian_gamma_occupation_integral(
            get_hydrogen_energy_difference(1, 0),
            FERMI_WAVEVECTOR["NICKEL"],
            Boltzmann * t,
        )
        for t in temperatures
    ]
    (line,) = ax.plot(temperatures, np.log(non_isf_ratio))
    line.set_label("2 band non isf")

    (line,) = ax.plot(
        temperatures,
        -get_hydrogen_energy_difference(0, 1) / (Boltzmann * temperatures),
    )
    line.set_label("Theoretical")

    ax.set_xlabel(r"T /k")
    ax.set_ylabel(r"$\lambda$")
    ax.legend()
    scale = FuncScale(ax, [lambda x: 1 / x, lambda x: 1 / x])
    ax.set_xscale(scale)
    fig.show()
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
