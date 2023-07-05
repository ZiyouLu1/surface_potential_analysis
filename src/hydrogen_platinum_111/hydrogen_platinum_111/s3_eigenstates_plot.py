from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from surface_potential_analysis.state_vector.eigenstate_collection_plot import (
    plot_energies_against_bloch_phase_1d,
)

from .s3_eigenstates import (
    get_eigenstate_collection_deuterium,
    get_eigenstate_collection_hydrogen,
)


def plot_lowest_band_energy_deuterium() -> None:
    fig, ax = plt.subplots()

    shapes = [
        (23, 23, 10),
        (23, 23, 12),
        (25, 25, 10),
        (27, 27, 10),
        (29, 29, 10),
        (29, 29, 12),
    ]
    for shape in shapes:
        collection = get_eigenstate_collection_deuterium(shape)
        _, _, ln = plot_energies_against_bloch_phase_1d(
            collection, np.array([1, 0, 0]), band=0, ax=ax
        )
        ln.set_label(f"({shape[0]}, {shape[1]}, {shape[2]})")

    ax.legend()
    fig.show()
    input()


def plot_lowest_band_energy_hydrogen() -> None:
    fig, ax = plt.subplots()

    shapes = [
        (23, 23, 10),
        (23, 23, 12),
        (25, 25, 10),
    ]
    for shape in shapes:
        collection = get_eigenstate_collection_hydrogen(shape)
        _, _, ln = plot_energies_against_bloch_phase_1d(
            collection, np.array([1, 0, 0]), band=0, ax=ax
        )
        ln.set_label(f"({shape[0]}, {shape[1]}, {shape[2]})")

    ax.legend()
    fig.show()
    input()


def plot_lowest_bands_hydrogen() -> None:
    fig, ax = plt.subplots()

    collection = get_eigenstate_collection_hydrogen((23, 23, 10))
    for band in range(8):
        plot_energies_against_bloch_phase_1d(
            collection, np.array([1, 0, 0]), band=band, ax=ax
        )

    print(  # noqa: T201
        np.min(collection["energies"], axis=0) - np.min(collection["energies"])
    )
    print(  # noqa: T201
        np.max(collection["energies"], axis=0) - np.min(collection["energies"])
    )
    print(  # noqa: T201
        np.max(collection["energies"], axis=0) - np.min(collection["energies"], axis=0)
    )

    fig.show()
    input()


def plot_lowest_bands_deuterium() -> None:
    fig, ax = plt.subplots()

    collection = get_eigenstate_collection_deuterium((29, 29, 10))
    for band in range(8):
        plot_energies_against_bloch_phase_1d(
            collection, np.array([1, 0, 0]), band=band, ax=ax
        )

    print(  # noqa: T201
        np.min(collection["energies"], axis=0) - np.min(collection["energies"])
    )
    print(  # noqa: T201
        np.max(collection["energies"], axis=0) - np.min(collection["energies"])
    )
    print(  # noqa: T201
        np.max(collection["energies"], axis=0) - np.min(collection["energies"], axis=0)
    )
    fig.show()
    input()
