from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from surface_potential_analysis.state_vector.eigenstate_collection_plot import (
    plot_eigenvalues_against_bloch_phase_1d,
)

from .s3_eigenstates import (
    get_eigenstate_collection_deuterium,
    get_eigenstate_collection_hydrogen,
)


def plot_lowest_band_energy_hydrogen() -> None:
    fig, ax = plt.subplots()

    shapes = [
        (25, 25, 10),
        (23, 23, 10),
        (23, 23, 12),
        (21, 21, 10),
    ]
    for shape in shapes:
        collection = get_eigenstate_collection_hydrogen(shape)
        _, _, ln = plot_eigenvalues_against_bloch_phase_1d(
            collection, np.array([1, 0, 0]), band=0, ax=ax
        )
        ln.set_label(f"({shape[0]}, {shape[1]}, {shape[2]})")

    ax.legend()
    fig.show()
    input()


def plot_lowest_bands_hydrogen() -> None:
    fig, ax = plt.subplots()

    collection = get_eigenstate_collection_hydrogen((25, 25, 10))
    for band in range(8):
        plot_eigenvalues_against_bloch_phase_1d(
            collection, np.array([1, 0, 0]), band=band, ax=ax
        )

    print(  # noqa: T201
        np.min(collection["eigenvalues"], axis=0) - np.min(collection["eigenvalues"])
    )

    fig.show()
    input()


def plot_lowest_band_energy_deuterium() -> None:
    fig, ax = plt.subplots()

    shapes = [
        (29, 29, 8),
        (31, 31, 8),
        (33, 33, 8),
        (33, 33, 10),
        (33, 33, 12),
        (35, 35, 8),
    ]
    for shape in shapes:
        collection = get_eigenstate_collection_deuterium(shape)
        _, _, ln = plot_eigenvalues_against_bloch_phase_1d(
            collection, np.array([1, 0, 0]), band=0, ax=ax
        )
        ln.set_label(f"({shape[0]}, {shape[1]}, {shape[2]})")

    ax.legend()
    fig.show()
    input()


def plot_lowest_bands_deuterium() -> None:
    fig, ax = plt.subplots()

    collection = get_eigenstate_collection_deuterium((25, 25, 10))
    for band in range(8):
        plot_eigenvalues_against_bloch_phase_1d(
            collection, np.array([1, 0, 0]), band=band, ax=ax
        )

    print(  # noqa: T201
        np.min(collection["eigenvalues"], axis=0) - np.min(collection["eigenvalues"])
    )

    fig.show()
    input()
