from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from surface_potential_analysis.state_vector.conversion import (
    interpolate_state_vector_momentum,
)
from surface_potential_analysis.state_vector.eigenstate_collection import (
    select_eigenstate,
)
from surface_potential_analysis.state_vector.eigenstate_collection_plot import (
    plot_energies_against_bloch_phase_1d,
    plot_lowest_band_energies_against_bloch_k,
    plot_occupation_against_bloch_phase_1d,
)
from surface_potential_analysis.state_vector.plot import (
    plot_state_vector_1d_x,
    plot_state_vector_difference_1d_k,
)

from sodium_copper_111.s1_potential_plot import plot_sodium_potential

from .s3_eigenstates import get_eigenstate_collection


def plot_lowest_band_energies() -> None:
    """Analyze the convergence of the eigenvalues."""
    fig, ax = plt.subplots()

    collection = get_eigenstate_collection((1000,))
    _, _, ln = plot_lowest_band_energies_against_bloch_k(collection, ax=ax)
    ln.set_label("(100)")

    collection = get_eigenstate_collection((2000,))
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


def plot_high_energy_band_eigenstates() -> None:
    fig, ax = plt.subplots()

    collection_0 = get_eigenstate_collection((1000,))

    for i in [0, 5]:
        eigenstate = select_eigenstate(collection_0, i, 16)
        _, _, ln = plot_state_vector_1d_x(eigenstate, ax=ax, measure="abs")
        ln.set_label(f"n={1000}")

    collection_1 = get_eigenstate_collection((5000,))

    for i in [0, 5]:
        eigenstate = select_eigenstate(collection_1, i, 16)
        eigenstate["vector"] *= np.sqrt(5)
        _, _, ln = plot_state_vector_1d_x(eigenstate, ax=ax, measure="abs")
        ln.set_label(f"n={5000}")

    ax2 = ax.twinx()
    _, _, ln = plot_sodium_potential((100,), ax=ax2)
    ln.set_linestyle("--")
    ln.set_label("potential")

    ax.legend()
    ax.set_title("Plot of eigenstates from the six lowest bands")

    fig.show()

    for i in [0, 5]:
        state_0 = interpolate_state_vector_momentum(
            select_eigenstate(collection_0, i, 16), (5000,)
        )
        state_0["vector"] *= np.exp(-1j * np.angle(state_0["vector"][0]))
        state_1 = select_eigenstate(collection_1, i, 16)
        state_1["vector"] *= np.exp(-1j * np.angle(state_1["vector"][0]))
        fig, _, _ = plot_state_vector_difference_1d_k(state_0, state_1)
        fig.show()
    input()


def plot_first_six_band_eigenstates() -> None:
    fig, ax = plt.subplots()

    collection = get_eigenstate_collection((1000,))

    for i in range(15, 17):
        eigenstate = select_eigenstate(collection, 0, i)
        _, _, ln = plot_state_vector_1d_x(eigenstate, ax=ax, measure="abs")
        ln.set_label(f"n={i}")

    ax2 = ax.twinx()
    _, _, ln = plot_sodium_potential((100,), ax=ax2)
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
