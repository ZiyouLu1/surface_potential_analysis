from __future__ import annotations

from surface_potential_analysis.potential.plot import (
    plot_potential_1d_x,
)

from .s1_potential import get_interpolated_potential


def plot_sodium_potential_100_point() -> None:
    potential = get_interpolated_potential((100,))
    fig, _, _ = plot_potential_1d_x(potential)
    fig.show()
    input()
