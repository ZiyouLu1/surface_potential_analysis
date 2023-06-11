from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from surface_potential_analysis.basis.conversion import (
    basis_as_fundamental_position_basis,
)
from surface_potential_analysis.potential.conversion import convert_potential_to_basis
from surface_potential_analysis.potential.plot import plot_potential_along_path
from surface_potential_analysis.state_vector.eigenstate_collection import (
    select_eigenstate,
)
from surface_potential_analysis.state_vector.eigenstate_collection_plot import (
    plot_energies_against_bloch_phase_1d,
    plot_lowest_band_energies_against_bloch_k,
    plot_occupation_against_bloch_phase_1d,
)
from surface_potential_analysis.state_vector.plot import plot_eigenstate_1d_x

from sodium_copper_111.s1_potential import get_interpolated_potential

from .s3_eigenstates import get_eigenstate_collection


def plot_lowest_band_energies() -> None:
    """Analyze the convergence of the eigenvalues."""
    fig, ax = plt.subplots()

    collection = get_eigenstate_collection((1000,))
    _, _, ln = plot_lowest_band_energies_against_bloch_k(collection, ax=ax)
    ln.set_label("(100)")

    collection = get_eigenstate_collection((1000,))
    _, _, ln = plot_lowest_band_energies_against_bloch_k(collection, ax=ax)
    ln.set_label("(200)")

    ax.legend()
    ax.set_title("Plot of lowest band energies\nshowing convergence for n=100")

    fig.show()
    input()


def plot_first_six_band_energies() -> None:
    fig, ax = plt.subplots()

    collection = get_eigenstate_collection((1000,))
    direction = np.array([1])

    for i in range(25):
        _, _, ln = plot_energies_against_bloch_phase_1d(collection, direction, i, ax=ax)
        ln.set_label(f"n={i}")

    ax.legend()
    ax.set_title("Plot of 6 lowest band energies")

    fig.show()
    input()


def plot_first_six_band_eigenstates() -> None:
    fig, ax = plt.subplots()

    collection = get_eigenstate_collection((1000,))

    for i in range(15, 17):
        eigenstate = select_eigenstate(collection, 0, i)
        _, _, ln = plot_eigenstate_1d_x(eigenstate, ax=ax, measure="abs")
        ln.set_label(f"n={i}")

    ax2 = ax.twinx()
    potential = get_interpolated_potential((100,))
    plot_basis = basis_as_fundamental_position_basis(potential["basis"])
    converted = convert_potential_to_basis(potential, plot_basis)
    path = np.arange(100).reshape(1, -1)
    _, _, ln = plot_potential_along_path(converted, path, ax=ax2)
    ln.set_linestyle("--")
    ln.set_label("potential")

    ax.legend()
    ax.set_title("Plot of eigenstates from the six lowest bands")

    fig.show()
    input()


def plot_first_six_band_boltzmann_occupation() -> None:
    fig, ax = plt.subplots()

    collection = get_eigenstate_collection((1000,))
    direction = np.array([1])
    temperature = 155
    for i in [0, 12]:
        _, _, ln = plot_occupation_against_bloch_phase_1d(
            collection, direction, temperature, i, ax=ax
        )
        ln.set_label(f"n={i}")

    ax.legend()
    ax.set_title(
        f"Plot of occupation against band at T={temperature}K,\n"
        "demonstrating high occupation for large bands"
    )

    fig.show()
    input()
