from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np
import scipy.optimize
from matplotlib import pyplot as plt
from matplotlib.scale import FuncScale
from scipy.constants import Boltzmann
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.dynamics.hermitian_gamma_integral import (
    calculate_hermitian_gamma_occupation_integral,
)
from surface_potential_analysis.dynamics.incoherent_propagation.isf import (
    calculate_equilibrium_initial_state_isf,
    calculate_equilibrium_state_averaged_isf,
    calculate_isf_at_times,
    get_rate_decomposition,
)
from surface_potential_analysis.dynamics.incoherent_propagation.plot import (
    plot_rate_decomposition_against_temperature,
)
from surface_potential_analysis.dynamics.incoherent_propagation.tunnelling_matrix import (
    get_a_matrix_from_jump_matrix,
    get_initial_pure_density_matrix_for_basis,
    get_tunnelling_m_matrix,
)
from surface_potential_analysis.dynamics.isf import (
    ISFFeyModelFit,
    fit_isf_to_double_exponential,
    fit_isf_to_fey_model_110,
    fit_isf_to_fey_model_112bar,
)
from surface_potential_analysis.dynamics.isf_plot import (
    plot_isf_4_variable_fit_against_time,
    plot_isf_against_time,
    plot_isf_fey_model_fit_110_against_time,
    plot_isf_fey_model_fit_112bar_against_time,
)
from surface_potential_analysis.util.constants import FERMI_WAVEVECTOR
from surface_potential_analysis.util.decorators import npy_cached

from hydrogen_nickel_111.surface_data import get_data_path

from .experimental_data import get_experiment_data
from .s4_wavepacket import get_hydrogen_energy_difference, get_wavepacket_hydrogen
from .s6_a_calculation import (
    calculate_gamma_potential_integral_hydrogen_diagonal,
    get_fey_jump_matrix_hydrogen,
    get_tunnelling_a_matrix_hydrogen,
)

if TYPE_CHECKING:
    from pathlib import Path

_L0Inv = TypeVar("_L0Inv", bound=int)


def get_jianding_isf_110() -> np.ndarray[tuple[Literal[2]], np.dtype[np.float64]]:
    basis = get_wavepacket_hydrogen(0)["basis"][1]

    util = BasisUtil(basis)
    dk = util.delta_x_stacked[0] / np.linalg.norm(util.delta_x_stacked[0])
    dk *= 2 / np.linalg.norm(util.delta_x_stacked[0])
    return dk[:2]  # type: ignore[no-any-return]


def get_jianding_isf_112bar() -> np.ndarray[tuple[Literal[2]], np.dtype[np.float64]]:
    basis = get_wavepacket_hydrogen(0)["basis"][1]

    util = BasisUtil(basis)
    dk_0_norm = util.delta_x_stacked[0] / np.linalg.norm(util.delta_x_stacked[0])
    dk = util.delta_x_stacked[1] - dk_0_norm * np.dot(
        dk_0_norm, util.delta_x_stacked[1]
    )
    dk /= np.linalg.norm(dk)
    dk *= 2 / np.linalg.norm(util.delta_x_stacked[0])
    return dk[:2]  # type: ignore[no-any-return]


def get_jianding_isf_diagonal() -> np.ndarray[tuple[Literal[2]], np.dtype[np.float64]]:
    basis = get_wavepacket_hydrogen(0)["basis"][1]

    util = BasisUtil(basis)
    dk = util.delta_x_stacked[0] + util.delta_x_stacked[1]
    dk /= np.linalg.norm(dk)
    dk *= 2 / np.linalg.norm(util.delta_x_stacked[0])
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
) -> np.ndarray[tuple[Literal[3], Literal[7]], np.dtype[np.float64]]:
    temperatures = np.array([100, 125, 150, 175, 200, 225, 250])
    times = [
        np.linspace(0, 4e-8, 2000),
        np.linspace(0, 15e-9, 2000),
        np.linspace(0, 4e-9, 2000),
        np.linspace(0, 15e-10, 2000),
        np.linspace(0, 8e-10, 2000),
        np.linspace(0, 4e-10, 2000),
        np.linspace(0, 4e-10, 2000),
    ]
    dk = get_jianding_isf_110()

    rates = np.zeros((2, temperatures.shape[0]))
    for i, (temperature, ts) in enumerate(zip(temperatures, times, strict=True)):
        a_matrix = get_tunnelling_a_matrix_hydrogen((25, 25), 6, temperature)
        m_matrix = get_tunnelling_m_matrix(a_matrix, n_bands)
        isf = calculate_equilibrium_state_averaged_isf(m_matrix, ts, dk)
        fit = fit_isf_to_fey_model_112bar(isf)
        fit1 = fit_isf_to_fey_model_110(isf)
        fit2 = fit_isf_to_double_exponential(isf)
        if plot:
            fig, ax, line = plot_isf_fey_model_fit_112bar_against_time(fit, ts)
            line.set_label("112bar")
            _, _, line = plot_isf_against_time(isf, ax=ax)
            line.set_label("ISF")
            _, _, line = plot_isf_fey_model_fit_110_against_time(fit1, ts, ax=ax)
            line.set_label("100")
            _, _, line = plot_isf_4_variable_fit_against_time(fit2, ts, ax=ax)
            line.set_linestyle("--")
            line.set_label("4 variable fit")
            f, s = extract_intrinsic_rates(fit2.fast_rate, fit2.slow_rate)
            _, _, line = plot_isf_fey_model_fit_110_against_time(
                ISFFeyModelFit(f, s), ts, ax=ax
            )
            line.set_linestyle("-.")
            ax.legend()
            fig.show()
            input()
        rates[0, i] = fit2.fast_rate
        rates[1, i] = fit2.slow_rate
    return np.array([temperatures, rates[0], rates[1]])  # type: ignore[no-any-return]


def extract_intrinsic_rates(fast_rate: float, slow_rate: float) -> tuple[float, float]:
    a_dk = 2

    def _func(x: tuple[float, float]) -> list[float]:
        nu, lam = x
        y = np.sqrt(lam**2 + 2 * lam * (8 * np.cos(np.sqrt(3)) + 1) / 9 + 1)
        z = np.sqrt(
            9 * lam**2
            + 16 * lam * np.cos(a_dk / 2) ** 2
            + 16 * lam * np.cos(a_dk / 2)
            - 14 * lam
            + 9
        )
        return [
            nu * (3 * lam + 3 + z) / (6 * lam) - fast_rate,
            nu * (3 * lam + 3 - z) / (6 * lam) - slow_rate,
        ]
        return [
            nu * (lam + 1 + y) / (2 * lam) - fast_rate,
            nu * (lam + 1 - y) / (2 * lam) - slow_rate,
        ]

    result, _detail, _, _ = scipy.optimize.fsolve(  # type: ignore unknown # cSpell: disable-line
        _func,
        [slow_rate, slow_rate / fast_rate],
        full_output=True,
        xtol=1e-15,  # cSpell: disable-line
    )
    fey_slow = float(result[0])  # type: ignore unknown
    fey_fast = float(result[0] / result[1])  # type: ignore unknown
    np.testing.assert_array_almost_equal(_func(result), 0.0, decimal=5)  # type: ignore unknown
    return fey_fast, fey_slow


def calculate_rates_hydrogen_tight_binding(
    temperatures: np.ndarray[tuple[_L0Inv], np.dtype[np.float64]],
) -> np.ndarray[tuple[Literal[2], _L0Inv], np.dtype[np.float64]]:
    prefactor = 3 * calculate_gamma_potential_integral_hydrogen_diagonal(0, 1, (0, 0))

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
    (fast_rates_fey, slow_rates_fey) = (list[float](), list[float]())
    for fast, slow in zip(fast_rates, slow_rates, strict=True):
        f, s = extract_intrinsic_rates(fast, slow)
        fast_rates_fey.append(f)
        slow_rates_fey.append(s)

    (line,) = ax.plot(temperatures, fast_rates)
    line.set_label("Fast Rate")
    (line,) = ax.plot(temperatures, slow_rates)
    line.set_label("Slow Rate")

    (line,) = ax.plot(temperatures, fast_rates_fey)
    line.set_label("Fast Rate fey")
    (line,) = ax.plot(temperatures, slow_rates_fey)
    line.set_label("Slow Rate fey")

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

    scale = FuncScale(ax, [lambda x: 1 / x, lambda x: 1 / x])  # type: ignore unknown
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
    scale = FuncScale(ax, [lambda x: 1 / x, lambda x: 1 / x])  # type: ignore unknown
    ax.set_xscale(scale)
    fig.show()
    input()


def plot_isf_hydrogen() -> None:
    a_matrix = get_tunnelling_a_matrix_hydrogen((25, 25), 6, 150)
    times = np.linspace(0, 90e-10, 1000)
    dk = get_jianding_isf_112bar()

    fig, ax = plt.subplots()

    m_matrix = get_tunnelling_m_matrix(a_matrix, 6)
    initial = get_initial_pure_density_matrix_for_basis(m_matrix["basis"][0])
    isf = calculate_isf_at_times(m_matrix, initial, times, dk)
    fig, _, _ = plot_isf_against_time(isf, ax=ax)

    isf_fit = fit_isf_to_double_exponential(isf)
    fig, _, _ = plot_isf_4_variable_fit_against_time(isf_fit, times, ax=ax)

    m_matrix = get_tunnelling_m_matrix(a_matrix, 2)
    initial = get_initial_pure_density_matrix_for_basis(m_matrix["basis"][0])
    isf = calculate_isf_at_times(m_matrix, initial, times, dk)
    fig, _, _ = plot_isf_against_time(isf, ax=ax)
    fig.show()
    input()


def plot_isf_hydrogen_fey_model() -> None:
    a_matrix = get_a_matrix_from_jump_matrix(get_fey_jump_matrix_hydrogen(), (12, 12))
    times = np.linspace(0, 90e-10, 1000)
    dk = get_jianding_isf_112bar()

    fig, ax = plt.subplots()

    m_matrix = get_tunnelling_m_matrix(a_matrix)
    isf = calculate_equilibrium_state_averaged_isf(m_matrix, times, dk)
    _, _, line = plot_isf_against_time(isf, ax=ax)
    line.set_label("ISF")

    isf = calculate_equilibrium_initial_state_isf(m_matrix, times, dk)
    _, _, line = plot_isf_against_time(isf, ax=ax)
    line.set_label("ISF")

    exponential_fit = fit_isf_to_double_exponential(isf)
    _, _, line = plot_isf_4_variable_fit_against_time(exponential_fit, times, ax=ax)
    line.set_label("Double exponential Model")
    line.set_linestyle("--")

    fey_fit = fit_isf_to_fey_model_112bar(isf)
    _, _, line = plot_isf_fey_model_fit_112bar_against_time(fey_fit, times, ax=ax)
    line.set_label("Fey Model 112")
    line.set_linestyle("--")

    fey_fit = fit_isf_to_fey_model_110(isf)
    _, _, line = plot_isf_fey_model_fit_110_against_time(fey_fit, times, ax=ax)
    line.set_label("Fey Model 110")
    line.set_linestyle("--")

    fey_fit = ISFFeyModelFit(3 * 6.55978349e08, 3 * 3.02959631e08)
    _, _, line = plot_isf_fey_model_fit_112bar_against_time(fey_fit, times, ax=ax)
    line.set_label("Fey Model 112 actual")
    line.set_linestyle("--")

    _, _, line = plot_isf_fey_model_fit_110_against_time(fey_fit, times, ax=ax)
    line.set_label("Fey Model 110 actual")
    line.set_linestyle("--")

    ax.legend()
    fig.show()
    input()


def compare_isf_initial_condition() -> None:
    a_matrix = get_tunnelling_a_matrix_hydrogen((25, 25), 6, 150)
    m_matrix = get_tunnelling_m_matrix(a_matrix)

    fig, ax = plt.subplots()
    dk = get_jianding_isf_112bar()

    initial = get_initial_pure_density_matrix_for_basis(m_matrix["basis"][0], (0, 0, 0))
    times = np.linspace(0, 90e-10, 1000)
    isf = calculate_isf_at_times(m_matrix, initial, times, dk)
    _, _, line = plot_isf_against_time(isf, ax=ax)
    line.set_label("Initially FCC")

    initial = get_initial_pure_density_matrix_for_basis(m_matrix["basis"][0], (0, 0, 1))
    isf = calculate_isf_at_times(m_matrix, initial, times, dk)
    _, _, line = plot_isf_against_time(isf, ax=ax)
    line.set_label("Initially HCP")

    ax.legend()
    fig.show()
    input()


def plot_relevant_rates_hydrogen() -> None:
    a_matrix = get_tunnelling_a_matrix_hydrogen((25, 25), 6, 150)
    m_matrix = get_tunnelling_m_matrix(a_matrix, n_bands=6)
    initial = get_initial_pure_density_matrix_for_basis(m_matrix["basis"][0], (0, 0, 0))
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

    scale = FuncScale(ax, [lambda x: 1 / x, lambda x: 1 / x])  # type: ignore unknown
    ax.set_xscale(scale)
    ax.set_yscale("log")

    m_matrix = get_tunnelling_m_matrix(a_matrix, n_bands=6)
    initial = get_initial_pure_density_matrix_for_basis(m_matrix["basis"][0], (0, 0, 1))
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
