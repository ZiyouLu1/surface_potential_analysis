from __future__ import annotations

import numpy as np
from surface_potential_analysis.axis.plot import plot_explicit_basis_states_x
from surface_potential_analysis.potential.plot import (
    plot_potential_1d_x,
)
from surface_potential_analysis.stacked_basis.potential_basis import (
    select_minimum_potential_3d,
)

from .s1_potential import (
    get_interpolated_potential,
)
from .s2_hamiltonian import (
    get_hamiltonian,
)


def plot_hydrogen_basis() -> None:
    shape = (50, 50, 250)
    hamiltonian = get_hamiltonian(
        shape=shape,
        bloch_fraction=np.array([0, 0, 0]),
        resolution=(3, 3, 12),
    )
    fig, ax0, _ = plot_explicit_basis_states_x(hamiltonian["basis"][0][2])

    potential = get_interpolated_potential(shape)
    minimum = select_minimum_potential_3d(potential)
    _, ax1, _ = plot_potential_1d_x(minimum, (0,), (), ax=ax0.twinx())  # type: ignore type is Axis

    ax1.set_ylim(0, 1e-18)
    ax0.set_ylim(0)

    fig.show()

    fig, _, _ = plot_explicit_basis_states_x(hamiltonian["basis"][0][2], measure="imag")
    fig.show()
    input()
