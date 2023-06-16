from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from surface_potential_analysis.potential.plot import plot_potential_along_path

from .s1_potential import get_interpolated_potential

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D


def plot_sodium_potential(
    shape: tuple[int], ax: Axes | None = None
) -> tuple[Figure, Axes, Line2D]:
    potential = get_interpolated_potential(shape)
    path = np.arange(shape[0]).reshape(1, -1)
    return plot_potential_along_path(potential, path, ax=ax)


def plot_sodium_potential_100_point() -> None:
    fig, _, _ = plot_sodium_potential((100,))
    fig.show()
    input()
