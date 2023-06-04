from __future__ import annotations

from typing import TypeVar, overload

import numpy as np
import scipy.integrate
from scipy.constants import Boltzmann, electron_mass, elementary_charge, epsilon_0, hbar

_S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])


@overload
def _get_delta_e(
    k_0: np.ndarray[_S0Inv, np.dtype[np.float_]],
    k_1: np.ndarray[_S0Inv, np.dtype[np.float_]] | float,
) -> np.ndarray[_S0Inv, np.dtype[np.float_]]:
    ...


@overload
def _get_delta_e(
    k_0: float,
    k_1: np.ndarray[_S0Inv, np.dtype[np.float_]],
) -> np.ndarray[_S0Inv, np.dtype[np.float_]] | float:
    ...


@overload
def _get_delta_e(
    k_0: float,
    k_1: float,
) -> float:
    ...


def _get_delta_e(
    k_0: np.ndarray[_S0Inv, np.dtype[np.float_]] | float,
    k_1: np.ndarray[_S0Inv, np.dtype[np.float_]] | float,
) -> np.ndarray[_S0Inv, np.dtype[np.float_]] | float:
    e_0 = (hbar * k_0) ** 2 / (2 * electron_mass)
    e_1 = (hbar * k_1) ** 2 / (2 * electron_mass)
    return e_0 - e_1  # type: ignore[no-any-return]


def _get_fermi_occupation(
    k: np.ndarray[_S0Inv, np.dtype[np.float_]], k_f: float, boltzmann_energy: float
) -> np.ndarray[_S0Inv, np.dtype[np.float_]]:
    return 1 / (1 + np.exp((_get_delta_e(k, k_f)) / boltzmann_energy))  # type: ignore[no-any-return]


def _actual_integrand(
    k_1: np.ndarray[_S0Inv, np.dtype[np.float_]],
    omega: float,
    k_f: float,
    boltzmann_energy: float,
) -> np.ndarray[_S0Inv, np.dtype[np.float_]]:
    k_3 = np.sqrt(k_1**2 - 2 * electron_mass * omega / (hbar**2))
    return (  # type: ignore[no-any-return]
        (k_1**2)
        * (k_3**2)
        * _get_fermi_occupation(k_1, k_f, boltzmann_energy)
        * (1 - _get_fermi_occupation(k_3, k_f, boltzmann_energy))
        / (np.sqrt(k_1**2 - 2 * electron_mass * omega / (hbar**2)))
    )


def _approximate_integrand(
    k_1: np.ndarray[_S0Inv, np.dtype[np.float_]],
    omega: float,
    k_f: float,
    boltzmann_energy: float,
) -> np.ndarray[_S0Inv, np.dtype[np.float_]]:
    return (  # type: ignore[no-any-return]
        (k_f**3)
        * 0.25
        * np.exp(-((_get_delta_e(k_1, k_f) / (2 * boltzmann_energy)) ** 2))
        * np.exp(-omega / (2 * boltzmann_energy))
    )


def calculate_approximate_electron_integral(
    fermi_k: float, energy_jump: float, temperature: float = 150
) -> float:
    """
    Calculate the approximate electron integral given energy_jump.

    Parameters
    ----------
    fermi_k : float
    energy_jump : float
    temperature : float, optional
        temperature, by default 150

    Returns
    -------
    float
    """
    boltzmann_energy = Boltzmann * temperature
    d_k = 2 * boltzmann_energy * electron_mass / (hbar**2 * fermi_k)

    return scipy.integrate.quad(  # type: ignore[no-any-return]
        lambda k: _approximate_integrand(k, energy_jump, fermi_k, boltzmann_energy),
        fermi_k - 20 * d_k,
        fermi_k + 20 * d_k,
    )[0]


def calculate_electron_integral(
    fermi_k: float, energy_jump: float, temperature: float = 150
) -> float:
    """
    Calculate the actual electron integral given energy_jump.

    Parameters
    ----------
    fermi_k : float
    energy_jump : float
    temperature : float, optional
        temperature, by default 150

    Returns
    -------
    float
    """
    boltzmann_energy = Boltzmann * temperature
    d_k = 2 * boltzmann_energy * electron_mass / (hbar**2 * fermi_k)

    integral = scipy.integrate.quad(
        lambda k: _actual_integrand(k, energy_jump, fermi_k, boltzmann_energy),
        fermi_k - 20 * d_k,
        fermi_k + 20 * d_k,
    )[0]
    prefactor = (
        128
        * (hbar / electron_mass)
        * (epsilon_0 * hbar**2 / (elementary_charge**2 * electron_mass)) ** 2
    )
    return prefactor * integral  # type: ignore[no-any-return]
