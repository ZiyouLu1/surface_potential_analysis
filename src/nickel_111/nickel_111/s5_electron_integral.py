from typing import TypeVar

import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import (
    Boltzmann,
    electron_mass,
    elementary_charge,
    epsilon_0,
    hbar,
    physical_constants,
)

bohr_radius = physical_constants["Bohr radius"][0]

fermi_wavevector_nickel = 1.77e10
fermi_energy_nickel = (hbar * fermi_wavevector_nickel) ** 2 / (2 * electron_mass)

_L0Inv = TypeVar("_L0Inv", bound=int)


def electron_occupation(
    energies: np.ndarray[tuple[_L0Inv], np.dtype[np.float_]], T: float
) -> np.ndarray[tuple[_L0Inv], np.dtype[np.float_]]:
    kt = Boltzmann * T
    e = np.array(energies)
    return 1 / (1 + np.exp((e - fermi_energy_nickel) / kt))  # type: ignore


def overlap_potential(
    dk_points: np.ndarray[tuple[_L0Inv], np.dtype[np.float_]]
) -> np.ndarray[tuple[_L0Inv], np.dtype[np.float_]]:
    """
    _summary_

    Parameters
    ----------
    dk_points : ArrayLike
        List of k points for the scattering vector dk (List of kx,ky,kz coordinates)

    Returns
    -------
    NDarray
        A list of the overlap potential at each point
    """

    alpha = 2 / bohr_radius
    print(alpha)

    q_points = np.linalg.norm(dk_points, axis=-1)

    prefactor = elementary_charge**2 / (epsilon_0 * q_points**2)
    aa = 1 / (1 + (q_points / alpha) ** 2) ** 2
    return prefactor * (aa - 1)  # type: ignore


def plot_electron_occupation() -> None:
    energies = np.linspace(fermi_energy_nickel * 0.99, fermi_energy_nickel * 1.01, 1000)
    fig, ax = plt.subplots()
    ax.plot(energies, electron_occupation(energies, 100))
    ax.set_title("Plot of the electron occupation in Nickel around the fermi surface")
    fig.show()

    qx_points = np.linspace(1e8, 6e10, 1000)
    dk_points = np.zeros(shape=(qx_points.shape[0], 3))
    dk_points[:, 0] = qx_points
    fig, ax = plt.subplots()
    ax.plot(qx_points, overlap_potential(dk_points))
    ax.set_title(
        "Plot of the Electron Hydrogen potential\n" "as a function of momentum transfer"
    )
    ax.set_ylim(None, 0)
    ax.set_xlim(0, None)
    fig.show()

    input()
