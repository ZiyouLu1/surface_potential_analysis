from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from surface_potential_analysis.state_vector.eigenstate_collection_plot import (
    plot_energies_against_bloch_phase_1d,
)

from .s3_eigenstates import get_eigenstate_collection


def plot_lowest_band_energy() -> None:
    fig, ax = plt.subplots()

    shapes = [
        (23, 23, 10),
        (25, 25, 10),
    ]
    for shape in shapes:
        collection = get_eigenstate_collection(shape)
        _, _, ln = plot_energies_against_bloch_phase_1d(
            collection, np.array([1, 0, 0]), band=0, ax=ax
        )
        ln.set_label(f"({shape[0]}, {shape[1]}, {shape[2]})")

    ax.legend()
    fig.show()
    input()


def plot_lowest_bands() -> None:
    fig, ax = plt.subplots()

    collection = get_eigenstate_collection((23, 23, 10))
    for band in range(3):
        plot_energies_against_bloch_phase_1d(
            collection, np.array([1, 0, 0]), band=band, ax=ax
        )

    ax.legend()

    fig.show()
    input()
