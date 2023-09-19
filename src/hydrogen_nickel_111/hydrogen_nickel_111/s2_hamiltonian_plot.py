from __future__ import annotations

from typing import Any

import numpy as np
from surface_potential_analysis.axis.plot import plot_explicit_basis_states_x
from surface_potential_analysis.potential.plot import (
    plot_potential_1d_x,
)
from surface_potential_analysis.potential.potential import (
    normalize_potential,
    truncate_potential,
)
from surface_potential_analysis.stacked_basis.potential_basis import (
    PotentialBasisConfig,
    get_potential_basis_config_eigenstates,
    select_minimum_potential_3d,
)
from surface_potential_analysis.state_vector.eigenstate_collection_plot import (
    plot_states_1d_x,
)
from surface_potential_analysis.util.constants import HYDROGEN_MASS

from hydrogen_nickel_111.s1_potential import (
    extrapolate_uneven_potential,
    get_interpolated_potential,
    get_raw_potential_reciprocal_grid,
    interpolate_nickel_potential,
)

from .s2_hamiltonian import (
    get_hamiltonian_deuterium,
    get_hamiltonian_hydrogen,
)


def plot_deuterium_basis() -> None:
    shape = (50, 50, 100)
    hamiltonian = get_hamiltonian_deuterium(
        shape=shape,
        bloch_fraction=np.array([0, 0, 0]),
        resolution=(2, 2, 12),
    )
    fig, ax, _ = plot_explicit_basis_states_x(hamiltonian["basis"][0][2])

    potential = get_interpolated_potential(shape)
    minimum = select_minimum_potential_3d(potential)
    _, _, _ = plot_potential_1d_x(minimum, (0,), (), ax=ax.twinx())

    fig.show()
    input()


def plot_hydrogen_basis() -> None:
    shape = (50, 50, 100)
    hamiltonian = get_hamiltonian_hydrogen(
        shape=shape,
        bloch_fraction=np.array([0, 0, 0]),
        resolution=(2, 2, 12),
    )
    fig, ax, _ = plot_explicit_basis_states_x(hamiltonian["basis"][0][2])

    potential = get_interpolated_potential(shape)
    minimum = select_minimum_potential_3d(potential)
    _, _, _ = plot_potential_1d_x(minimum, (0,), (), ax=ax.twinx())

    fig.show()
    input()


def plot_hydrogen_basis_extrapolated() -> None:
    shape = (50, 50, 250)
    potential = normalize_potential(get_raw_potential_reciprocal_grid())
    potential = extrapolate_uneven_potential(potential)
    interpolated = interpolate_nickel_potential(potential, shape)
    interpolated["data"] = 0.5 * (
        interpolated["data"]
        + interpolated["data"].reshape(shape).swapaxes(0, 1).ravel()
    )
    config: PotentialBasisConfig[Any, Any] = {
        "n": 16,
        "mass": HYDROGEN_MASS,
        "potential": select_minimum_potential_3d(interpolated),
    }
    states = get_potential_basis_config_eigenstates(config)

    fig, _ = plot_states_1d_x(states)
    fig.show()
    fig, _ = plot_states_1d_x(states, measure="imag")
    fig.show()
    fig, _ = plot_states_1d_x(states, measure="angle")
    fig.show()

    potential = truncate_potential(potential, cutoff=3.5e-19, n=5, offset=1e-20)
    interpolated = interpolate_nickel_potential(potential, shape)
    interpolated["data"] = 0.5 * (
        interpolated["data"]
        + interpolated["data"].reshape(shape).swapaxes(0, 1).ravel()
    )
    config_1: PotentialBasisConfig[Any, Any] = {
        "n": 16,
        "mass": HYDROGEN_MASS,
        "potential": select_minimum_potential_3d(interpolated),
    }
    states = get_potential_basis_config_eigenstates(config_1)

    fig, _ = plot_states_1d_x(states)
    fig.show()
    fig, _ = plot_states_1d_x(states, measure="imag")
    fig.show()
    fig, _ = plot_states_1d_x(states, measure="angle")
    fig.show()
    input()
