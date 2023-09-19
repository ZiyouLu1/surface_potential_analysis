from __future__ import annotations

from typing import TypeVar

import numpy as np
import scipy.constants
from matplotlib import pyplot as plt
from surface_potential_analysis.wavepacket.plot import (
    plot_wavepacket_eigenvalues_2d_k,
    plot_wavepacket_eigenvalues_2d_x,
)

from .s4_wavepacket import get_wavepacket_hydrogen
from .surface_data import save_figure


def plot_copper_wavepacket_eigenvalues() -> None:
    for i in range(10):
        wavepacket = get_wavepacket_hydrogen(i)
        fig, _, _ = plot_wavepacket_eigenvalues_2d_k(wavepacket)
        fig.show()

        fig, _, _ = plot_wavepacket_eigenvalues_2d_x(wavepacket)
        fig.show()
    input()


def get_copper_eigenvalues() -> np.ndarray[tuple[int, int, int], np.dtype[np.float_]]:
    out = np.zeros((10, 12, 12))
    for i in range(10):
        wavepacket = get_wavepacket_hydrogen(i)
        out[i] = wavepacket["eigenvalue"]
    return out  # type: ignore[no-any-return]


def plot_copper_bands_occupation() -> None:
    eigenvalues = get_copper_eigenvalues()[:, 0, 0]
    # Plot the eigenstate occupation. Need to think about there 'mu' is
    # i.e. we have more than one hydrogen adsorbed on the surface
    # And interaction between hydrogen would also ruin things
    fig, ax = plt.subplots()

    ax.plot(eigenvalues, np.exp(-eigenvalues / (scipy.constants.Boltzmann * 60)))
    ax.set_xlabel("Eigenvalue Index")
    ax.set_ylabel("Occupation Probability")
    ax.set_title(
        "Plot of occupation probability of each band according to the Boltzmann distribution"
    )
    fig.show()
    save_figure(fig, "copper_bands_occupation.png")


def list_first_copper_band_with_widths() -> None:
    print("----------------------------------------")  # noqa: T201
    print("Not relaxed data")  # noqa: T201

    eigenvalues = get_copper_eigenvalues()
    center_eigenvalues = eigenvalues[:, 0, 0]
    max_eigenvalues = np.max(eigenvalues, axis=0)
    print("k=(0,0)")  # noqa: T201
    print(np.subtract(center_eigenvalues, center_eigenvalues[0])[:5])  # noqa: T201
    print("k=(max, max)")  # noqa: T201
    print(np.subtract(max_eigenvalues, center_eigenvalues[0])[:5])  # noqa: T201
    print("bandwidths")  # noqa: T201
    print(np.subtract(max_eigenvalues, center_eigenvalues)[:5])  # noqa: T201

    print("----------------------------------------")  # noqa: T201


def find_band_with_1mev_bandwidth() -> None:
    """
    Activated tunnelling has an energy of 197meV.

    John: The key thing is not the physical barrier,
    but the energy wrt the ground state of the first band with a decent (eg 1meV) bandwidth
    """
    print("----------------------------------------")  # noqa: T201
    print("Relaxed data")  # noqa: T201

    eigenvalues = get_copper_eigenvalues()
    center_eigenvalues = eigenvalues[:, 0, 0]
    max_eigenvalues = np.max(eigenvalues, axis=0)

    bandwidths = np.subtract(max_eigenvalues, center_eigenvalues)
    first_relevant = int(
        np.argmax(bandwidths > 1 * 10**-3 * scipy.constants.elementary_charge)
    )

    print("band index", first_relevant)  # noqa: T201
    print("band width", bandwidths[first_relevant])  # noqa: T201
    print(  # noqa: T201
        "k=0", center_eigenvalues[first_relevant] - center_eigenvalues[0]
    )
    print("bandwidths", bandwidths[: first_relevant + 3])  # noqa: T201
    eigenvalues = np.subtract(center_eigenvalues, center_eigenvalues[0])
    print("eigenvalues", eigenvalues[: first_relevant + 3])  # noqa: T201

    print("----------------------------------------")  # noqa: T201


def find_band_with_relevant_energy() -> None:
    """Activated tunnelling has an energy of 197meV - which band would this correspond to?."""
    print("----------------------------------------")  # noqa: T201
    print("Relaxed data")  # noqa: T201

    eigenvalues = get_copper_eigenvalues()
    eigenvalues_origin = eigenvalues[:, 0, 0]

    eigenvalues_max = np.max(eigenvalues, axis=0)

    bandwidths = np.abs(eigenvalues_origin - eigenvalues_max)
    first_relevant = int(
        np.argmax(bandwidths > 180 * 10**-3 * scipy.constants.elementary_charge)
    )
    last_relevant = int(
        np.argmax(bandwidths > 200 * 10**-3 * scipy.constants.elementary_charge)
    )

    print("band index", first_relevant, last_relevant)  # noqa: T201
    print("band width", bandwidths[first_relevant : last_relevant + 1])  # noqa: T201
    print(  # noqa: T201
        "k=0",
        eigenvalues_origin[first_relevant : last_relevant + 1] - eigenvalues_origin[0],
    )
    print(bandwidths[first_relevant : last_relevant + 1])  # noqa: T201

    print("----------------------------------------")  # noqa: T201


_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)


def calculate_bandwidths(
    eigenvalues: np.ndarray[tuple[_L0Inv, _L1Inv, _L2Inv], np.dtype[np.float_]]
) -> np.ndarray[tuple[_L0Inv], np.dtype[np.float_]]:
    return np.abs(np.subtract(np.max(eigenvalues, axis=0), np.min(eigenvalues, axis=0)))  # type: ignore[no-any-return]


def calculate_tst_rate_contributions(
    temperature: float,
    eigenvalues: np.ndarray[tuple[_L0Inv, _L1Inv, _L2Inv], np.dtype[np.float_]],
) -> np.ndarray[tuple[_L0Inv], np.dtype[np.float_]]:
    bandwidths = calculate_bandwidths(eigenvalues)
    frequencies = bandwidths / (scipy.constants.hbar * np.pi)
    eigenvalues = np.array(eigenvalues["center"])
    tunnelling_contributions = np.exp(
        -eigenvalues / (scipy.constants.Boltzmann * temperature)
    )
    normalization: float = np.sum(tunnelling_contributions)
    return (frequencies * tunnelling_contributions) / normalization  # type: ignore[no-any-return]


def calculate_tst_rate(
    temperature: float,
    eigenvalues: np.ndarray[tuple[_L0Inv, _L1Inv, _L2Inv], np.dtype[np.float_]],
) -> float:
    return np.sum(calculate_tst_rate_contributions(temperature, eigenvalues))


def plot_tst_rate_arrhenius() -> None:
    eigenvalues = get_copper_eigenvalues()

    temperatures = np.linspace(60, 150, 20)
    rates = [calculate_tst_rate(t, eigenvalues) for t in temperatures]

    fig, ax = plt.subplots()
    log_rate = np.log(rates)
    inverse_temp = 1000 / temperatures
    ax.plot(inverse_temp, log_rate)
    fig.show()
    ax.set_xlabel(r"$\frac{1000}{T}$")
    ax.set_ylabel(r"Hop rate $s^{-1}$")
    ax.set_yscale("log")

    rate_120k = calculate_tst_rate(120, eigenvalues)
    rate_190k = calculate_tst_rate(190, eigenvalues)
    # E / K * (1/T1 - 1/T2) =
    difference_rate = np.log(rate_120k) - np.log(rate_190k)
    difference_t = ((1 / 120) - (1 / 190)) / scipy.constants.Boltzmann
    energy = difference_rate / difference_t
    print(energy)  # noqa: T201

    print(np.argmax(calculate_tst_rate_contributions(130, eigenvalues)))  # noqa: T201
    input()
