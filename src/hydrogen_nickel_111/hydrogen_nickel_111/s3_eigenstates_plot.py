from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from surface_potential_analysis.axis.plot import plot_explicit_basis_states_x
from surface_potential_analysis.state_vector.conversion import (
    interpolate_state_vector_momentum,
)
from surface_potential_analysis.state_vector.eigenstate_collection import (
    load_eigenstate_collection,
    select_eigenstate,
)
from surface_potential_analysis.state_vector.eigenstate_collection_plot import (
    plot_energies_against_bloch_phase_1d,
    plot_lowest_band_energies_against_bloch_k,
)
from surface_potential_analysis.state_vector.plot import (
    animate_eigenstate_x0x1,
    plot_state_vector_difference_2d_k,
    plot_state_vector_difference_2d_x,
)

from hydrogen_nickel_111.s3_eigenstates import (
    get_eigenstate_collection_deuterium,
    get_eigenstate_collection_hydrogen,
)

from .surface_data import get_data_path, save_figure


def plot_deuterium_lowest_bands() -> None:
    fig, ax = plt.subplots()

    collection = get_eigenstate_collection_deuterium((23, 23, 12))
    for band in range(3):
        _, _, ln = plot_energies_against_bloch_phase_1d(
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
        (23, 23, 10),
        (23, 23, 12),
        (24, 24, 12),
        (25, 25, 12),
        (27, 27, 10),
        (28, 28, 8),
        (29, 29, 8),
    ]
    for shape in shapes:
        collection = get_eigenstate_collection_deuterium(shape)
        _, _, ln = plot_energies_against_bloch_phase_1d(
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
        (23, 23, 12),
        (24, 24, 10),
        (24, 24, 12),
        (25, 25, 10),
        (27, 27, 10),
        (29, 29, 10),
    ]
    for shape in shapes:
        collection = get_eigenstate_collection_hydrogen(shape)
        _, _, ln = plot_energies_against_bloch_phase_1d(
            collection, np.array([1, 0, 0]), band=0, ax=ax
        )
        ln.set_label(f"({shape[0]}, {shape[1]}, {shape[2]})")

    ax.legend(loc="lower right")
    fig.show()
    input()


def plot_hydrogen_lowest_band_energies() -> None:
    fig, ax = plt.subplots()

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    collection = get_eigenstate_collection_hydrogen((24, 24, 10))
    for band in range(8):
        _, _, ln = plot_energies_against_bloch_phase_1d(
            collection, np.array([1, 0, 0]), band=band, ax=ax
        )
        ln.set_label(f"band={band}")
        ln.set_color(colors[0])

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


def analyze_band_convergence() -> None:
    """Analyze the convergence of the eigenvalues."""
    fig, ax = plt.subplots()

    path = get_data_path("eigenstates_23_23_10.npy")
    eigenstates = load_eigenstate_collection(path)
    _, _, ln = plot_lowest_band_energies_against_bloch_k(eigenstates, ax=ax)
    ln.set_label("(23,23,10)")

    path = get_data_path("eigenstates_23_23_12.npy")
    eigenstates = load_eigenstate_collection(path)
    _, _, ln = plot_lowest_band_energies_against_bloch_k(eigenstates, ax=ax)
    ln.set_label("(23,23,12)")

    path = get_data_path("eigenstates_25_25_16.npy")
    eigenstates = load_eigenstate_collection(path)
    _, _, ln = plot_lowest_band_energies_against_bloch_k(eigenstates, ax=ax)
    ln.set_label("(25,25,16)")

    path = get_data_path("eigenstates_23_23_10_large.npy")
    eigenstates = load_eigenstate_collection(path)
    _, _, ln = plot_lowest_band_energies_against_bloch_k(eigenstates, ax=ax)
    ln.set_label("(23,23,10)")

    path = get_data_path("eigenstates_23_23_12_large.npy")
    eigenstates = load_eigenstate_collection(path)
    _, _, ln = plot_lowest_band_energies_against_bloch_k(eigenstates, ax=ax)
    ln.set_label("(23,23,12)")

    path = get_data_path("eigenstates_25_25_16_large.npy")
    eigenstates = load_eigenstate_collection(path)
    _, _, ln = plot_lowest_band_energies_against_bloch_k(eigenstates, ax=ax)
    ln.set_label("(25,25,16)")

    ax.legend()
    ax.set_title(
        "Plot of lowest band energies\n"
        "showing convergence for an eigenstate grid of (15,15,12)"
    )

    fig.show()
    save_figure(fig, "lowest_band_convergence.png")


def plot_sho_basis_states() -> None:
    """Plot the basis states used to generate the eigenvectors."""
    path = get_data_path("eigenstates_25_25_16.npy")
    collection = load_eigenstate_collection(path)

    eigenstate = select_eigenstate(collection, 0, 0)
    fig, _, _ = plot_explicit_basis_states_x(eigenstate["basis"][2], measure="real")
    fig.show()
    input()


def plot_eigenstate_for_each_band() -> None:
    """Check to see if the eigenstates look as they are supposed to."""
    path = get_data_path("eigenstates_25_25_16.npy")
    collection = load_eigenstate_collection(path)

    eigenstate = select_eigenstate(collection, 0, 0)
    fig, _, _anim0 = animate_eigenstate_x0x1(eigenstate)
    fig.show()

    eigenstate = select_eigenstate(collection, 0, 1)
    fig, _, _anim1 = animate_eigenstate_x0x1(eigenstate)
    fig.show()

    eigenstate = select_eigenstate(collection, 0, 2)
    fig, _, _anim2 = animate_eigenstate_x0x1(eigenstate)
    fig.show()

    eigenstate = select_eigenstate(collection, 0, 3)
    fig, _, _anim3 = animate_eigenstate_x0x1(eigenstate)
    fig.show()
