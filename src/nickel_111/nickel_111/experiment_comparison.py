import json
from typing import Literal, TypedDict

import numpy as np
import scipy.optimize
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.container import ErrorbarContainer
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy.constants import Boltzmann, electron_mass, elementary_charge, epsilon_0, hbar

from nickel_111.surface_data import get_data_path


class TunnellRateVariables(TypedDict):
    hcp_energy: float
    overlap: float
    fermi_k: float


def tunnelling_rate_constant():
    a = (hbar / elementary_charge**2) ** 2
    b = hbar * epsilon_0**2 / electron_mass**2
    c = np.sqrt(np.pi) * 32
    return a * b * c


def tunnelling_rate_nickel_prefactor(variables: TunnellRateVariables):
    return (
        variables["overlap"] ** 2
        * variables["fermi_k"] ** 2
        * tunnelling_rate_constant()
    )


def tunnelling_rate_nickel(
    temperatures: np.ndarray[tuple[int], np.dtype[np.float_]],
    variables: TunnellRateVariables,
) -> np.ndarray[tuple[int], np.dtype[np.float_]]:
    prefactor = tunnelling_rate_nickel_prefactor(variables)
    omega = variables["hcp_energy"]
    beta = 1 / (temperatures * Boltzmann)
    temperature_dependance = 2 * np.cosh(beta * omega / 2) / beta
    # 4 * print(temperature_dependance, prefactor, prefactor * temperature_dependance)
    return 3 * prefactor * temperature_dependance


class ExperimentData(TypedDict):
    temperature: np.ndarray[tuple[int], np.dtype[np.float_]]
    rate: np.ndarray[tuple[int], np.dtype[np.float_]]
    upper_error: np.ndarray[tuple[int], np.dtype[np.float_]]
    lower_error: np.ndarray[tuple[int], np.dtype[np.float_]]


def load_experiment_data() -> ExperimentData:
    path = get_data_path("fast_rate_experimental.json")
    with path.open("r") as f:
        out = json.load(f)
        return {
            "temperature": np.array(out["temperature"]),
            "lower_error": np.array(out["lower_error"]) * 10**10,  # type:ignore
            "rate": np.array(out["rate"]) * 10**10,  # type:ignore
            "upper_error": np.array(out["upper_error"]) * 10**10,  # type:ignore
        }


nickel_rate_variables: TunnellRateVariables = {
    "fermi_k": 1.77e10,
    "hcp_energy": 2.03e-21,  # 2.708825773687628e-21,
    "overlap": 0.0044,  # 4.1e-3,
}


def get_experimental_subtracted_rate(factor=1) -> ExperimentData:
    data = load_experiment_data()
    theoretical_rates = factor * tunnelling_rate_nickel(
        data["temperature"], nickel_rate_variables
    )
    return {
        "temperature": data["temperature"],
        "lower_error": data["lower_error"] - theoretical_rates,
        "rate": data["rate"] - theoretical_rates,
        "upper_error": data["upper_error"] - theoretical_rates,
    }


def get_experimental_baseline_rates(factor=1):
    subtracted = get_experimental_subtracted_rate(factor=factor)
    temperatures = subtracted["temperature"]
    rate = subtracted["rate"]

    def f(t, A, B):
        return B * np.exp(-A / t)

    A0 = np.log(rate[0] / rate[9]) / ((1 / temperatures[9]) - (1 / temperatures[0]))
    R0 = rate[0] * np.exp(A0 / temperatures[0])

    popt, pcov = scipy.optimize.curve_fit(f, temperatures, rate, p0=[A0, R0])
    print(A0, R0)
    print(popt, rate[0], f(temperatures[0], *popt))
    return lambda t: f(t, *popt)


def plot_tunnelling_rate_theory(
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    temperatures = np.linspace(100, 300)
    rates = tunnelling_rate_nickel(temperatures, nickel_rate_variables)
    rates += get_experimental_baseline_rates()(temperatures)

    (line,) = ax.plot(temperatures, rates)
    return fig, ax, line


def plot_tunnelling_rate_theory_double(
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    temperatures = np.linspace(100, 160)
    rates = 3 * tunnelling_rate_nickel(temperatures, nickel_rate_variables)
    rates += get_experimental_baseline_rates(factor=3)(temperatures)

    (line,) = ax.plot(temperatures, rates)
    return fig, ax, line


def plot_tunnelling_rate_baseline(
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    temperatures = np.linspace(100, 300)
    rates = get_experimental_baseline_rates()(temperatures)

    (line,) = ax.plot(temperatures, rates)
    return fig, ax, line


def plot_tunnelling_rate_jianding(
    ax: Axes | None = None,
) -> tuple[Figure, Axes, ErrorbarContainer]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    data = load_experiment_data()

    container = ax.errorbar(
        data["temperature"],
        data["rate"],
        yerr=[data["rate"] - data["lower_error"], data["upper_error"] - data["rate"]],
    )
    return fig, ax, container


def tunnelling_rate_nickel_constants(
    temperatures: np.ndarray[tuple[int], np.dtype[np.float_]],
    variables: TunnellRateVariables,
) -> np.ndarray[tuple[Literal[2], int], np.dtype[np.float_]]:
    prefactor = tunnelling_rate_nickel_prefactor(variables)
    omega = variables["hcp_energy"]
    beta = 1 / (temperatures * Boltzmann)
    # 4 * print(temperature_dependance, prefactor, prefactor * temperature_dependance)
    return (prefactor / beta) * np.array(
        [np.exp(beta * omega / 2), np.exp(-beta * omega / 2)]
    )


def calculate_rate_150K():
    print(tunnelling_rate_nickel_constants(np.array([150]), nickel_rate_variables))
