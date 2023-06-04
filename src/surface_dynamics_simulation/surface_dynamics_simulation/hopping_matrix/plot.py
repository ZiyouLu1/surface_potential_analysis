from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import Boltzmann, electron_mass, hbar

from .old_electron_integral import _actual_integrand, _approximate_integrand


def plot_electron_integral(
    fermi_k: float, energy_jump: float, temperature: float = 150
) -> None:
    """
    Plot the electron integrand.

    Parameters
    ----------
    fermi_k : float
    energy_jump : float
    temperature : float, optional
        temperature, by default 150
    """
    boltzmann_energy = Boltzmann * temperature
    d_k = 2 * boltzmann_energy * electron_mass / (hbar**2 * fermi_k)

    k_points = np.linspace(fermi_k - 2 * d_k, fermi_k + 8 * d_k, 1000)

    fig, ax = plt.subplots(1)
    ax.plot(
        k_points,
        _actual_integrand(k_points, energy_jump, fermi_k, boltzmann_energy),
        label="actual",
    )
    ax.plot(
        k_points,
        _approximate_integrand(k_points, energy_jump, fermi_k, boltzmann_energy),
        label="approximate",
    )
    ax.set_title(
        "Plot of actual integrand against k,\nwith the real value of $\\omega{}$"
    )
    ax.legend()
    ax.set_xlabel("Wavevector $m^{-1}$")
    fig.show()

    fig, ax = plt.subplots(1)
    ax.plot(
        k_points,
        _actual_integrand(k_points, 0, fermi_k, boltzmann_energy),
        label="actual",
    )
    ax.plot(
        k_points,
        _approximate_integrand(k_points, 0, fermi_k, boltzmann_energy),
        label="approximate",
    )
    ax.set_title("Plot of actual integrand against k\nwith $\\omega{}=0$")
    ax.legend()
    ax.set_xlabel("Wavevector $m^{-1}$")
    fig.show()
    input()
