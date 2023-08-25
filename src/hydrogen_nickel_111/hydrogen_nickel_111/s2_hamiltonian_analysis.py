from __future__ import annotations

import numpy as np
from surface_potential_analysis.axis.axis import ExplicitAxis3d
from surface_potential_analysis.axis.plot import plot_explicit_basis_states_x
from surface_potential_analysis.basis.potential_basis import select_minimum_potential_3d
from surface_potential_analysis.potential.plot import (
    plot_potential_1d_x,
)

from hydrogen_nickel_111.s1_potential import get_interpolated_potential

from .s2_hamiltonian import (
    get_hamiltonian_deuterium,
    get_hamiltonian_hydrogen,
)


def _normalize_sho_basis(basis: ExplicitAxis3d[int, int]) -> ExplicitAxis3d[int, int]:
    turning_point = basis.vectors[
        np.arange(basis.vectors.shape[0]),
        np.argmax(np.abs(basis.vectors[:, : basis.vectors.shape[1] // 2]), axis=1),
    ]

    normalized = np.exp(-1j * np.angle(turning_point))[:, np.newaxis] * basis.vectors
    return ExplicitAxis3d(basis.delta_x, normalized)


def plot_deuterium_basis() -> None:
    shape = (50, 50, 100)
    hamiltonian = get_hamiltonian_deuterium(
        shape=shape,
        bloch_fraction=np.array([0, 0, 0]),
        resolution=(2, 2, 12),
    )
    fig, ax, _ = plot_explicit_basis_states_x(hamiltonian["basis"][2])

    potential = get_interpolated_potential(shape)
    minimum = select_minimum_potential_3d(potential)
    _, _, _ = plot_potential_1d_x(minimum, 0, (), ax=ax.twinx())

    fig.show()
    input()


def plot_hydrogen_basis() -> None:
    shape = (50, 50, 100)
    hamiltonian = get_hamiltonian_hydrogen(
        shape=shape,
        bloch_fraction=np.array([0, 0, 0]),
        resolution=(2, 2, 12),
    )
    fig, ax, _ = plot_explicit_basis_states_x(hamiltonian["basis"][2])

    potential = get_interpolated_potential(shape)
    minimum = select_minimum_potential_3d(potential)
    _, _, _ = plot_potential_1d_x(minimum, 0, (), ax=ax.twinx())

    fig.show()
    input()
