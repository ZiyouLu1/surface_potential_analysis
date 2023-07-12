from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from surface_potential_analysis.state_vector.eigenstate_collection_plot import (
    plot_eigenvalues_against_bloch_phase_1d,
)

from .s3_eigenstates import get_eigenstate_collection, get_eigenstate_collection_relaxed

PLOT_COLOURS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def plot_lowest_bands_comparison() -> None:
    fig, ax = plt.subplots()

    shape = (23, 23, 14)
    for band in range(5):
        collection = get_eigenstate_collection(shape)
        collection["eigenvalues"] -= np.min(collection["eigenvalues"])
        _, _, ln = plot_eigenvalues_against_bloch_phase_1d(
            collection, np.array([1, 0, 0]), band=band, ax=ax
        )
        ln.set_color(PLOT_COLOURS[band])

    for band in range(5):
        collection = get_eigenstate_collection_relaxed(shape)
        collection["eigenvalues"] -= np.min(collection["eigenvalues"])
        _, _, ln = plot_eigenvalues_against_bloch_phase_1d(
            collection, np.array([1, 0, 0]), band=band, ax=ax
        )
        ln.set_color(PLOT_COLOURS[band])
        ln.set_linestyle("--")

    fig.show()
    input()
