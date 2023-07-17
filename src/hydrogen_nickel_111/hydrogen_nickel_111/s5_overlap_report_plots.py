from __future__ import annotations

from typing import TypeVar

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.scale import FuncScale
from scipy.constants import Boltzmann, electron_mass, hbar
from surface_potential_analysis.basis.util import AxisWithLengthBasisUtil
from surface_potential_analysis.dynamics.hermitian_gamma_integral import (
    calculate_hermitian_gamma_occupation_integral,
    calculate_hermitian_gamma_potential_integral,
    get_hermitian_gamma_occupation_integrand,
)
from surface_potential_analysis.overlap.conversion import (
    convert_overlap_to_momentum_basis,
)
from surface_potential_analysis.overlap.interpolation import (
    get_overlap_momentum_interpolator_flat,
)
from surface_potential_analysis.overlap.plot import plot_overlap_2d_k, plot_overlap_2d_x
from surface_potential_analysis.util.constants import FERMI_WAVEVECTOR

from hydrogen_nickel_111.experimental_data import (
    get_experiment_data,
    get_experimental_baseline_rates,
)
from hydrogen_nickel_111.s4_wavepacket import (
    get_deuterium_energy_difference,
    get_hydrogen_energy_difference,
)
from hydrogen_nickel_111.s5_overlap import (
    get_overlap_deuterium,
    get_overlap_hydrogen,
)
from hydrogen_nickel_111.s5_overlap_analysis import get_angle_averaged_overlap

_S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])

PLOT_COLOURS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def plot_fermi_occupation_intregrand() -> None:
    fig, ax = plt.subplots()

    dk = 2 * Boltzmann * 100 * electron_mass / (hbar**2 * FERMI_WAVEVECTOR["NICKEL"])
    k1_points = np.linspace(
        FERMI_WAVEVECTOR["NICKEL"] - 5 * dk, FERMI_WAVEVECTOR["NICKEL"] + 5 * dk
    )

    for i, t in enumerate([150, 170, 190]):
        occupation = get_hermitian_gamma_occupation_integrand(
            k1_points,
            omega=0,
            k_f=FERMI_WAVEVECTOR["NICKEL"],
            boltzmann_energy=Boltzmann * t,
        )
        (line,) = ax.plot(k1_points, occupation)
        line.set_label(f"{t} K")
        line.set_color(PLOT_COLOURS[i])

        omega = float(get_hydrogen_energy_difference(0, 1))
        occupation = get_hermitian_gamma_occupation_integrand(
            k1_points,
            omega=omega,
            k_f=FERMI_WAVEVECTOR["NICKEL"],
            boltzmann_energy=Boltzmann * t,
        )
        (line,) = ax.plot(k1_points, occupation)
        line.set_linestyle("--")
        line.set_color(PLOT_COLOURS[i])

        occupation = get_hermitian_gamma_occupation_integrand(
            k1_points,
            omega=-omega,
            k_f=FERMI_WAVEVECTOR["NICKEL"],
            boltzmann_energy=Boltzmann * t,
        )
        (line,) = ax.plot(k1_points, occupation)
        line.set_linestyle("--")
        line.set_color(PLOT_COLOURS[i])

    line = ax.axvline(FERMI_WAVEVECTOR["NICKEL"])
    line.set_linestyle("--")
    line.set_alpha(0.5)
    line.set_color("black")

    ax.set_ylim(0, None)
    ax.set_xlim(k1_points[0], k1_points[-1])
    ax.set_xlabel("$k_1$ / $m$")
    ax.set_ylabel("$n_{1}(1-n_{3})$")
    ax.legend()
    fig.show()
    input()


def plot_fermi_occupation_integral() -> None:
    fig, ax = plt.subplots()

    temperatures = np.linspace(100, 250, 100)
    integrals_0_offset = [
        calculate_hermitian_gamma_occupation_integral(
            0, FERMI_WAVEVECTOR["NICKEL"], Boltzmann * t
        )
        for t in temperatures
    ]
    omega = float(get_hydrogen_energy_difference(0, 1))
    integrals_non_zero_offset = [
        calculate_hermitian_gamma_occupation_integral(
            omega, FERMI_WAVEVECTOR["NICKEL"], Boltzmann * t
        )
        for t in temperatures
    ]

    integrals_non_zero_offset2 = [
        calculate_hermitian_gamma_occupation_integral(
            -omega, FERMI_WAVEVECTOR["NICKEL"], Boltzmann * t
        )
        for t in temperatures
    ]

    (line,) = ax.plot(1 / temperatures, integrals_0_offset)
    line.set_label("$\\omega$ = $0$")

    (line,) = ax.plot(1 / temperatures, integrals_non_zero_offset)
    line.set_label("$\\omega$ = $\\omega_{0,1}$")

    (line,) = ax.plot(1 / temperatures, integrals_non_zero_offset2)
    line.set_label("$\\omega$ = $\\omega_{1,0}$")

    ax.set_xlabel("1 / $T$")
    ax.set_ylabel("integral $n_{1}(1-n_{3})$")
    ax.set_xlim(1 / temperatures[-1], 1 / temperatures[0])
    ax.legend()
    fig.show()
    input()


def get_hydrogen_fcc_hcp_gamma() -> np.complex_:
    overlap = get_overlap_hydrogen(0, 1)
    interpolator_0_1 = get_overlap_momentum_interpolator_flat(overlap)

    def overlap_function(
        q: np.ndarray[_S0Inv, np.dtype[np.float_]]
    ) -> np.ndarray[_S0Inv, np.dtype[np.complex_]]:
        return get_angle_averaged_overlap(
            interpolator_0_1, interpolator_0_1, q.ravel()
        ).reshape(q.shape)

    return calculate_hermitian_gamma_potential_integral(
        FERMI_WAVEVECTOR["NICKEL"], overlap_function
    )


def get_rate_simple_equation_hydrogen(
    temperature: np.ndarray[_S0Inv, np.dtype[np.float_]]
) -> np.ndarray[_S0Inv, np.dtype[np.float_]]:
    omega = float(get_hydrogen_energy_difference(0, 1))
    temperature_flat = temperature.ravel()
    temperature_dep_integral = np.array(
        [
            calculate_hermitian_gamma_occupation_integral(
                omega, FERMI_WAVEVECTOR["NICKEL"], Boltzmann * t
            )
            for t in temperature_flat
        ]
    )
    temperature_dep_integral2 = np.array(
        [
            calculate_hermitian_gamma_occupation_integral(
                -omega, FERMI_WAVEVECTOR["NICKEL"], Boltzmann * t
            )
            for t in temperature_flat
        ]
    )
    fcc_hcp_gamma = 13.465247231353736  # get_hydrogen_fcc_hcp_gamma()
    return (  # type: ignore[no-any-return]
        (temperature_dep_integral + temperature_dep_integral2) * (3 * (fcc_hcp_gamma))
    ).reshape(temperature.shape)


def get_deuterium_fcc_hcp_gamma() -> np.complex_:
    overlap = get_overlap_deuterium(0, 1)
    interpolator_0_1 = get_overlap_momentum_interpolator_flat(overlap)

    def overlap_function(
        q: np.ndarray[_S0Inv, np.dtype[np.float_]]
    ) -> np.ndarray[_S0Inv, np.dtype[np.complex_]]:
        return get_angle_averaged_overlap(
            interpolator_0_1, interpolator_0_1, q.ravel()
        ).reshape(q.shape)

    return calculate_hermitian_gamma_potential_integral(
        FERMI_WAVEVECTOR["NICKEL"], overlap_function
    )


def get_rate_simple_equation_deuterium(
    temperature: np.ndarray[_S0Inv, np.dtype[np.float_]]
) -> np.ndarray[_S0Inv, np.dtype[np.float_]]:
    omega = float(get_deuterium_energy_difference(0, 1))
    temperature_flat = temperature.ravel()
    temperature_dep_integral = np.array(
        [
            calculate_hermitian_gamma_occupation_integral(
                omega, FERMI_WAVEVECTOR["NICKEL"], Boltzmann * t
            )
            for t in temperature_flat
        ]
    )
    temperature_dep_integral2 = np.array(
        [
            calculate_hermitian_gamma_occupation_integral(
                -omega, FERMI_WAVEVECTOR["NICKEL"], Boltzmann * t
            )
            for t in temperature_flat
        ]
    )
    fcc_hcp_gamma = 0.012759755043446671  #!get_deuterium_fcc_hcp_gamma()
    return (  # type: ignore[no-any-return]
        (temperature_dep_integral + temperature_dep_integral2) * (3 * (fcc_hcp_gamma))
    ).reshape(temperature.shape)


def plot_rate_equation() -> None:
    fig, ax = plt.subplots()

    temperatures = np.linspace(100, 300)
    rates = get_rate_simple_equation_hydrogen(temperatures)
    rates += get_experimental_baseline_rates(get_rate_simple_equation_hydrogen)(
        temperatures
    )
    (theory_line,) = ax.plot(temperatures, rates)

    data = get_experiment_data()
    exp_data_plot = ax.errorbar(
        data["temperature"],
        data["rate"],
        yerr=[data["rate"] - data["lower_error"], data["upper_error"] - data["rate"]],
    )
    rates_deuterium = get_rate_simple_equation_deuterium(temperatures)
    ax.plot(temperatures, rates_deuterium)

    scale = FuncScale(ax, [lambda x: 1 / x, lambda x: 1 / x])
    ax.set_xscale(scale)
    ax.set_yscale("log")
    ax.legend([theory_line, exp_data_plot], ["theoretical rate", "experimental rate"])
    ax.set_ylim(5e8, None)
    ax.set_ylabel("Rate /s")
    ax.set_xlabel("1/Temperature 1/$\\mathrm{K}^{-1}$")

    fig.show()
    input()


def plot_overlap_2d_comparison() -> None:
    overlap = get_overlap_hydrogen(0, 0)
    overlap_momentum = convert_overlap_to_momentum_basis(overlap)
    util = AxisWithLengthBasisUtil(overlap["basis"])
    k1_max = util.get_stacked_index(np.argmax(overlap["vector"]))[1]
    fig, ax, _ = plot_overlap_2d_k(overlap_momentum, (0, 2), (k1_max,), measure="abs")
    fig.show()
    ax.set_xlim(-2e11, 2e11)
    ax.set_ylim(-2e11, 2e11)
    input()

    overlap = get_overlap_hydrogen(0, 0)
    overlap_momentum = convert_overlap_to_momentum_basis(overlap)
    fig, ax, _ = plot_overlap_2d_k(overlap_momentum, (1, 0), (0,), measure="abs")
    fig.show()
    ax.set_xlim(-2e11, 2e11)
    ax.set_ylim(-1.5e11, 1.5e11)
    input()

    overlap = get_overlap_hydrogen(0, 1)
    overlap_momentum = convert_overlap_to_momentum_basis(overlap)
    fig, ax, _ = plot_overlap_2d_k(overlap_momentum, (1, 0), (0,), measure="abs")
    fig.show()
    ax.set_xlim(-2e11, 2e11)
    ax.set_ylim(-1.5e11, 1.5e11)
    input()

    overlap = get_overlap_hydrogen(0, 1, (1, 1), (1, 1))
    util = AxisWithLengthBasisUtil(overlap["basis"])
    x2_max = util.get_stacked_index(np.argmax(overlap["vector"]))[2]

    fig, ax, _ = plot_overlap_2d_x(overlap, (0, 1), (x2_max,), measure="abs")
    ax.set_xlim(3e-10, 6e-10)
    ax.set_ylim(-0.5e-10, 2e-10)
    fig.show()
    input()
