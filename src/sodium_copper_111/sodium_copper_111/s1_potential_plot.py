from __future__ import annotations

import numpy as np
from surface_potential_analysis.potential.plot import plot_potential_along_path

from .s1_potential import get_interpolated_potential


def plot_sodium_potential() -> None:
    shape = (100,)
    potential = get_interpolated_potential(shape)

    path = np.arange(shape[0]).reshape(1, -1)
    fig, _, _ = plot_potential_along_path(potential, path)
    fig.show()
    input()
