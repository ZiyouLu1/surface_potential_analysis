from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar

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
    fit_isf_to_fey_model,
    get_rate_decomposition,
)
from surface_potential_analysis.dynamics.incoherent_propagation.isf_plot import (
    plot_isf_4_variable_fit_against_time,
    plot_isf_against_time,
    plot_isf_fey_model_fit_against_time,
    plot_rate_decomposition_against_temperature,
)
from surface_potential_analysis.dynamics.incoherent_propagation.tunnelling_matrix import (
    get_initial_pure_density_matrix_for_basis,
    get_tunnelling_m_matrix,
)
from surface_potential_analysis.util.constants import FERMI_WAVEVECTOR
from surface_potential_analysis.util.decorators import npy_cached

from hydrogen_nickel_111.surface_data import get_data_path

from .experimental_data import get_experiment_data
from .s4_wavepacket import get_hydrogen_energy_difference, get_wavepacket_hydrogen
from .s6_a_calculation import (
    calculate_gamma_potential_integral_hydrogen_diagonal,
    get_tunnelling_a_matrix_hydrogen,
)

if TYPE_CHECKING:
    from pathlib import Path

_L0Inv = TypeVar("_L0Inv", bound=int)


def get_jianding_isf_dk() -> np.ndarray[tuple[Literal[2]], np.dtype[np.float_]]:
    basis = get_wavepacket_hydrogen(0)["basis"]

    util = AxisWithLengthBasisUtil(basis)
    dk = util.delta_x[0] + util.delta_x[1]
    dk /= np.linalg.norm(dk)
    dk *= 0.8 * 10**10
    return dk[:2]  # type: ignore[no-any-return]


def _calculate_rates_hydrogen_cache(
    n_bands: int = 6,
    *,
    plot: bool = False,  # noqa: ARG001
) -> Path:
    return get_data_path(f"dynamics/rates_hydrogen_{n_bands}_band.npy")


@npy_cached(_calculate_rates_hydrogen_cache)
def calculate_rates_hydrogen(
    n_bands: int = 6,
    *,
    plot: bool = False,
) -> np.ndarray[tuple[Literal[3], Literal[7]], np.dtype[np.float_]]:
    temperatures = np.array([100, 125, 150, 175, 200, 225, 250])
    times = [
        np.linspace(0, 5e-8, 2000),
        np.linspace(0, 15e-9, 2000),
        np.linspace(0, 4e-9, 2000),
        np.linspace(0, 15e-10, 2000),
        np.linspace(0, 8e-10, 2000),
        np.linspace(0, 4e-10, 2000),
        np.linspace(0, 4e-10, 2000),
    ]
    dk = get_jianding_isf_dk()

    rates = np.zeros((2, temperatures.shape[0]))
    for i, (temperature, ts) in enumerate(zip(temperatures, times, strict=True)):
        a_matrix = get_tunnelling_a_matrix_hydrogen((25, 25), 6, temperature)
        m_matrix = get_tunnelling_m_matrix(a_matrix, n_bands)
        isf = calculate_equilibrium_state_averaged_isf(m_matrix, ts, dk)
        fit = fit_isf_to_fey_model(isf, ts)
        if plot:
            fig, ax, _ = plot_isf_fey_model_fit_against_time(fit, ts)
            plot_isf_against_time(isf, ts, ax=ax)
            fig.show()
            input(fit)
        rates[0, i] = fit.fast_rate
        rates[1, i] = fit.slow_rate
    return np.array([temperatures, rates[0], rates[1]])  # type: ignore[no-any-return]


def calculate_rates_hydrogen_tight_binding(
    temperatures: np.ndarray[tuple[_L0Inv], np.dtype[np.float_]],
) -> np.ndarray[tuple[Literal[2], _L0Inv], np.dtype[np.float_]]:
    prefactor = 3 * calculate_gamma_potential_integral_hydrogen_diagonal(
        0, 1, (0, 0), (0, 0)
    )

    fast_rates = [
        prefactor
        * calculate_hermitian_gamma_occupation_integral(
            get_hydrogen_energy_difference(1, 0),
            FERMI_WAVEVECTOR["NICKEL"],
            Boltzmann * t,
        )
        for t in temperatures
    ]
    slow_rates = [
        prefactor
        * calculate_hermitian_gamma_occupation_integral(
            get_hydrogen_energy_difference(0, 1),
            FERMI_WAVEVECTOR["NICKEL"],
            Boltzmann * t,
        )
        for t in temperatures
    ]

    return np.array([fast_rates, slow_rates])  # type: ignore[no-any-return]


def plot_tunnelling_rate_hydrogen() -> None:
    fig, ax = plt.subplots()

    temperatures, fast_rates, slow_rates = calculate_rates_hydrogen(plot=True)
    (line,) = ax.plot(temperatures, fast_rates)
    line.set_label("Fast Rate")
    (line,) = ax.plot(temperatures, slow_rates)
    line.set_label("Slow Rate")

    temperatures, fast_rates, slow_rates = calculate_rates_hydrogen(
        n_bands=2, plot=True
    )
    (line,) = ax.plot(temperatures, fast_rates)
    line.set_label("Fast Rate 2 band")
    (line,) = ax.plot(temperatures, slow_rates)
    line.set_label("Slow Rate 2 band")

    fast_rates, slow_rates = calculate_rates_hydrogen_tight_binding(temperatures)
    (line,) = ax.plot(temperatures, fast_rates)
    line.set_label("Fast Rate tb")
    (line,) = ax.plot(temperatures, slow_rates)
    line.set_label("Slow Rate tb")

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
    fig, ax = plt.subplots()

    temperatures, fast_rates, slow_rates = calculate_rates_hydrogen()
    (line,) = ax.plot(temperatures, np.log(fast_rates / slow_rates))
    line.set_linestyle("")
    line.set_marker("x")
    line.set_label("6 band")

    temperatures, fast_rates, slow_rates = calculate_rates_hydrogen(n_bands=2)
    (line,) = ax.plot(temperatures, np.log(fast_rates / slow_rates))
    line.set_linestyle("")
    line.set_marker("x")
    line.set_label("2 band")

    non_isf_ratio = [
        calculate_hermitian_gamma_occupation_integral(
            get_hydrogen_energy_difference(1, 0),
            FERMI_WAVEVECTOR["NICKEL"],
            Boltzmann * t,
        )
        / calculate_hermitian_gamma_occupation_integral(
            get_hydrogen_energy_difference(0, 1),
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
    a_matrix = get_tunnelling_a_matrix_hydrogen((25, 25), 6, 150)
    times = np.linspace(0, 90e-10, 1000)
    dk = get_jianding_isf_dk()

    fig, ax = plt.subplots()

    m_matrix = get_tunnelling_m_matrix(a_matrix, 6)
    initial = get_initial_pure_density_matrix_for_basis(m_matrix["basis"])
    isf = calculate_isf_at_times(m_matrix, initial, times, dk)
    fig, _, _ = plot_isf_against_time(isf, times, ax=ax)

    isf_fit = fit_isf_to_double_exponential(isf, times)
    fig, _, _ = plot_isf_4_variable_fit_against_time(isf_fit, times, ax=ax)

    m_matrix = get_tunnelling_m_matrix(a_matrix, 2)
    initial = get_initial_pure_density_matrix_for_basis(m_matrix["basis"])
    isf = calculate_isf_at_times(m_matrix, initial, times, dk)
    fig, _, _ = plot_isf_against_time(isf, times, ax=ax)
    fig.show()
    input()


def compare_isf_initial_condition() -> None:
    a_matrix = get_tunnelling_a_matrix_hydrogen((25, 25), 6, 150)
    m_matrix = get_tunnelling_m_matrix(a_matrix)

    fig, ax = plt.subplots()
    dk = get_jianding_isf_dk()

    initial = get_initial_pure_density_matrix_for_basis(m_matrix["basis"], (0, 0, 0))
    times = np.linspace(0, 90e-10, 1000)
    isf = calculate_isf_at_times(m_matrix, initial, times, dk)
    _, _, line = plot_isf_against_time(isf, times, ax=ax)
    line.set_label("Initially FCC")

    initial = get_initial_pure_density_matrix_for_basis(m_matrix["basis"], (0, 0, 1))
    isf = calculate_isf_at_times(m_matrix, initial, times, dk)
    _, _, line = plot_isf_against_time(isf, times, ax=ax)
    line.set_label("Initially HCP")

    ax.legend()
    fig.show()
    input()


def plot_relevant_rates_hydrogen() -> None:
    a_matrix = get_tunnelling_a_matrix_hydrogen((25, 25), 6, 150)
    m_matrix = get_tunnelling_m_matrix(a_matrix, n_bands=6)
    initial = get_initial_pure_density_matrix_for_basis(m_matrix["basis"], (0, 0, 0))
    temperatures = np.array([100, 125, 150, 175, 200, 225, 250])
    rates = [
        get_rate_decomposition(
            get_tunnelling_m_matrix(
                get_tunnelling_a_matrix_hydrogen((25, 25), 6, t), n_bands=6
            ),
            initial,
        )
        for t in temperatures
    ]
    fig, ax = plot_rate_decomposition_against_temperature(rates, temperatures)

    data = get_experiment_data()
    ax.errorbar(
        data["temperature"],
        data["rate"],
        yerr=[data["rate"] - data["lower_error"], data["upper_error"] - data["rate"]],
    )

    scale = FuncScale(ax, [lambda x: 1 / x, lambda x: 1 / x])
    ax.set_xscale(scale)
    ax.set_yscale("log")

    m_matrix = get_tunnelling_m_matrix(a_matrix, n_bands=6)
    initial = get_initial_pure_density_matrix_for_basis(m_matrix["basis"], (0, 0, 1))
    temperatures = np.array([100, 125, 150, 175, 200, 225, 250])
    rates = [
        get_rate_decomposition(
            get_tunnelling_m_matrix(
                get_tunnelling_a_matrix_hydrogen((25, 25), 6, t), n_bands=6
            ),
            initial,
        )
        for t in temperatures
    ]
    plot_rate_decomposition_against_temperature(rates, temperatures, ax=ax)
    fig.show()
    input()
