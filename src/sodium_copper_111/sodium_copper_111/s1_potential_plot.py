from __future__ import annotations

import numpy as np
from surface_potential_analysis.basis.conversion import (
    basis_as_fundamental_position_basis,
)
from surface_potential_analysis.potential.conversion import convert_potential_to_basis
from surface_potential_analysis.potential.plot import plot_potential_along_path

from .s1_potential import get_interpolated_potential


def plot_sodium_potential() -> None:
    potential = get_interpolated_potential((100,))
    plot_basis = basis_as_fundamental_position_basis(potential["basis"])
    converted = convert_potential_to_basis(potential, plot_basis)

    path = np.arange(100).reshape(1, -1)
    fig, _, _ = plot_potential_along_path(converted, path)
    fig.show()
    input()
