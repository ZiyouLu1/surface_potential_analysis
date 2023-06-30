from __future__ import annotations

from typing import TypeVar

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.scale import FuncScale
from scipy.constants import Boltzmann, electron_mass, hbar
from surface_potential_analysis.dynamics.hermitian_gamma_integral import (
    calculate_hermitian_gamma_occupation_integral,
    get_hermitian_gamma_occupation_integrand,
)

from hydrogen_nickel_111.experimental_data import (
    get_experiment_data,
    get_experimental_baseline_rates,
)
from hydrogen_nickel_111.s5_overlap import get_fcc_hcp_energy_difference

from .constants import FERMI_WAVEVECTOR

_S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])

PLOT_COLOURS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def plot_fermi_occupation_intregrand() -> None:
    fig, ax = plt.subplots()

    dk = 2 * Boltzmann * 100 * electron_mass / (hbar**2 * FERMI_WAVEVECTOR)
    k1_points = np.linspace(FERMI_WAVEVECTOR - 5 * dk, FERMI_WAVEVECTOR + 5 * dk)

    for i, t in enumerate([150, 170, 190]):
        occupation = get_hermitian_gamma_occupation_integrand(
            k1_points, omega=0, k_f=FERMI_WAVEVECTOR, boltzmann_energy=Boltzmann * t
        )
        (line,) = ax.plot(k1_points, occupation)
        line.set_label(f"{t} K")
        line.set_color(PLOT_COLOURS[i])

        omega = float(get_fcc_hcp_energy_difference())
        occupation = get_hermitian_gamma_occupation_integrand(
            k1_points, omega=omega, k_f=FERMI_WAVEVECTOR, boltzmann_energy=Boltzmann * t
        )
        (line,) = ax.plot(k1_points, occupation)
        line.set_linestyle("--")
        line.set_color(PLOT_COLOURS[i])

        occupation = get_hermitian_gamma_occupation_integrand(
            k1_points,
            omega=-omega,
            k_f=FERMI_WAVEVECTOR,
            boltzmann_energy=Boltzmann * t,
        )
        (line,) = ax.plot(k1_points, occupation)
        line.set_linestyle("--")
        line.set_color(PLOT_COLOURS[i])

    line = ax.axvline(FERMI_WAVEVECTOR)
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
            0, FERMI_WAVEVECTOR, Boltzmann * t
        )
        for t in temperatures
    ]
    omega = float(get_fcc_hcp_energy_difference())
    integrals_non_zero_offset = [
        calculate_hermitian_gamma_occupation_integral(
            omega, FERMI_WAVEVECTOR, Boltzmann * t
        )
        for t in temperatures
    ]

    integrals_non_zero_offset2 = [
        calculate_hermitian_gamma_occupation_integral(
            -omega, FERMI_WAVEVECTOR, Boltzmann * t
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


def get_rate_simple_equation(
    temperature: np.ndarray[_S0Inv, np.dtype[np.float_]]
) -> np.ndarray[_S0Inv, np.dtype[np.float_]]:
    omega = float(get_fcc_hcp_energy_difference())
    temperature_flat = temperature.ravel()
    temperature_dep_integral = np.array(
        [
            calculate_hermitian_gamma_occupation_integral(
                omega, FERMI_WAVEVECTOR, Boltzmann * t
            )
            for t in temperature_flat
        ]
    )
    temperature_dep_integral2 = np.array(
        [
            calculate_hermitian_gamma_occupation_integral(
                -omega, FERMI_WAVEVECTOR, Boltzmann * t
            )
            for t in temperature_flat
        ]
    )
    fcc_hcp_gamma = 26.93
    return (
        (temperature_dep_integral + temperature_dep_integral2) * (3 * (fcc_hcp_gamma))
    ).reshape(temperature.shape)


def plot_rate_equation() -> None:
    fig, ax = plt.subplots()

    temperatures = np.linspace(100, 300)
    rates = get_rate_simple_equation(temperatures)
    rates += get_experimental_baseline_rates(get_rate_simple_equation)(temperatures)
    (line,) = ax.plot(temperatures, rates)

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
    ax.set_ylim(5e8, None)
    ax.set_ylabel("Rate /s")
    ax.set_xlabel("1/Temperature 1/$\\mathrm{K}^{-1}$")

    fig.show()
    input()
