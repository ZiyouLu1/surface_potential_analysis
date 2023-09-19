from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from surface_potential_analysis.state_vector.eigenstate_collection_plot import (
    plot_eigenvalues_against_bloch_phase_1d,
)

from hydrogen_copper_111.s3_eigenstates import (
    get_eigenstate_collection_deuterium,
    get_eigenstate_collection_hydrogen,
)


def plot_lowest_band_energy_deuterium() -> None:
    shapes = [
        (21, 21, 8),
        (21, 21, 10),
        (21, 21, 12),
        (21, 21, 14),
        (21, 21, 16),
        (21, 21, 18),
        (21, 21, 20),
        (29, 29, 22),
        (29, 29, 24),
        (29, 29, 26),
        (27, 27, 7),
        (29, 29, 7),
        (31, 31, 7),
    ]
    for shape in shapes:
        get_eigenstate_collection_deuterium(shape)


def plot_lowest_band_energy_hydrogen() -> None:
    fig, ax = plt.subplots()

    shapes = [
        (23, 23, 12),
        (23, 23, 13),
        (23, 23, 14),
        (25, 25, 14),
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
