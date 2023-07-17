from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from surface_potential_analysis.axis.plot import plot_explicit_basis_states_x
from surface_potential_analysis.state_vector.conversion import (
    interpolate_state_vector_momentum,
)
from surface_potential_analysis.state_vector.eigenstate_collection import (
    select_eigenstate,
)
from surface_potential_analysis.state_vector.eigenstate_collection_plot import (
    plot_eigenvalues_against_bloch_phase_1d,
)
from surface_potential_analysis.state_vector.plot import (
    plot_state_vector_difference_2d_k,
    plot_state_vector_difference_2d_x,
)

from hydrogen_nickel_111.s3_eigenstates import (
    get_eigenstate_collection_deuterium,
    get_eigenstate_collection_hydrogen,
    get_eigenstate_collection_hydrogen_sho,
)


def plot_deuterium_lowest_bands() -> None:
    fig, ax = plt.subplots()

    collection = get_eigenstate_collection_deuterium((23, 23, 12))
    for band in range(3):
        _, _, ln = plot_eigenvalues_against_bloch_phase_1d(
            collection, np.array([1, 0, 0]), band=band, ax=ax
        )
        ln.set_label("(23, 23, 12)")

    ax.legend()
    ax.set_title("Plot of lowest band energies\nshowing convergence for n=100")

    fig.show()
    input()


def plot_deuterium_lowest_band_energy() -> None:
    fig, ax = plt.subplots()

    shapes = [
        (25, 25, 8),
        (27, 27, 8),
        (29, 29, 8),
        (31, 31, 8),
    ]
    for shape in shapes:
        collection = get_eigenstate_collection_deuterium(shape)
        _, _, ln = plot_eigenvalues_against_bloch_phase_1d(
            collection, np.array([1, 0, 0]), band=0, ax=ax
        )
        ln.set_label(f"({shape[0]}, {shape[1]}, {shape[2]})")

    ax.legend()
    fig.show()
    input()


def plot_hydrogen_lowest_band_energy() -> None:
    fig, ax = plt.subplots()

    shapes = [
        (23, 23, 10),
        (25, 25, 10),
        (27, 27, 10),
        (27, 27, 12),
        (29, 29, 10),
    ]
    for shape in shapes:
        collection = get_eigenstate_collection_hydrogen(shape)
        _, _, ln = plot_eigenvalues_against_bloch_phase_1d(
            collection, np.array([1, 0, 0]), band=0, ax=ax
        )
        ln.set_label(f"({shape[0]}, {shape[1]}, {shape[2]})")

    ax.legend(loc="lower right")
    fig.show()
    input()


def plot_hydrogen_lowest_bands() -> None:
    fig, ax = plt.subplots()

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    collection = get_eigenstate_collection_hydrogen((24, 24, 10))
    collection["eigenvalues"] -= np.min(collection["eigenvalues"])
    for band in range(8):
        _, _, ln = plot_eigenvalues_against_bloch_phase_1d(
            collection, np.array([1, 0, 0]), band=band, ax=ax
        )
        ln.set_label(f"band={band}")
        ln.set_color(colors[0])

    collection = get_eigenstate_collection_hydrogen_sho((24, 24, 10))
    collection["eigenvalues"] -= np.min(collection["eigenvalues"])
    for band in range(8):
        _, _, ln = plot_eigenvalues_against_bloch_phase_1d(
            collection, np.array([1, 0, 0]), band=band, ax=ax
        )
        ln.set_label(f"band={band}")
        ln.set_color(colors[1])

    ax.legend()
    ax.set_title("Plot of eight lowest band energies")

    fig.show()
    input()


def plot_state_vector_difference_hydrogen() -> None:
    collection_0 = get_eigenstate_collection_hydrogen((24, 24, 12))
    collection_1 = get_eigenstate_collection_hydrogen((26, 26, 12))

    state_0 = interpolate_state_vector_momentum(
        select_eigenstate(collection_0, 0, 0), (26, 26), (0, 1)
    )
    state_0["vector"] *= np.exp(-1j * np.angle(state_0["vector"][0]))
    state_1 = select_eigenstate(collection_1, 0, 0)
    state_1["vector"] *= np.exp(-1j * np.angle(state_1["vector"][0]))

    fig, _, _ = plot_state_vector_difference_2d_k(state_0, state_1, (1, 0))
    fig.show()

    z_max = np.argmax(collection_1["basis"][2].vectors[0])
    fig, _, _ = plot_state_vector_difference_2d_x(
        state_0, state_1, axes=(0, 1), idx=(z_max,)
    )
    fig.show()
    input()


def plot_sho_basis_states() -> None:
    """Plot the basis states used to generate the eigenvectors."""
    collection = get_eigenstate_collection_hydrogen_sho((25, 25, 16))

    eigenstate = select_eigenstate(collection, 0, 0)
    fig, _, _ = plot_explicit_basis_states_x(eigenstate["basis"][2], measure="real")
    fig.show()
    input()
