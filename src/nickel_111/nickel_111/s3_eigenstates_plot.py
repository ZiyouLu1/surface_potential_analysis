from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from surface_potential_analysis.basis.plot import plot_explicit_basis_states_x
from surface_potential_analysis.eigenstate.eigenstate_collection import (
    load_eigenstate_collection,
    select_eigenstate,
)
from surface_potential_analysis.eigenstate.eigenstate_collection_plot import (
    plot_energies_against_bloch_phase_1d,
    plot_lowest_band_energies_against_bloch_kx,
)
from surface_potential_analysis.eigenstate.plot import animate_eigenstate_x0x1

from .surface_data import get_data_path, save_figure


def analyze_band_convergence() -> None:
    """Analyze the convergence of the eigenvalues."""
    fig, ax = plt.subplots()

    path = get_data_path("eigenstates_23_23_10.npy")
    eigenstates = load_eigenstate_collection(path)
    _, _, ln = plot_lowest_band_energies_against_bloch_kx(eigenstates, ax=ax)
    ln.set_label("(23,23,10)")

    path = get_data_path("eigenstates_23_23_12.npy")
    eigenstates = load_eigenstate_collection(path)
    _, _, ln = plot_lowest_band_energies_against_bloch_kx(eigenstates, ax=ax)
    ln.set_label("(23,23,12)")

    path = get_data_path("eigenstates_25_25_16.npy")
    eigenstates = load_eigenstate_collection(path)
    _, _, ln = plot_lowest_band_energies_against_bloch_kx(eigenstates, ax=ax)
    ln.set_label("(25,25,16)")

    path = get_data_path("eigenstates_23_23_10_large.npy")
    eigenstates = load_eigenstate_collection(path)
    _, _, ln = plot_lowest_band_energies_against_bloch_kx(eigenstates, ax=ax)
    ln.set_label("(23,23,10)")

    path = get_data_path("eigenstates_23_23_12_large.npy")
    eigenstates = load_eigenstate_collection(path)
    _, _, ln = plot_lowest_band_energies_against_bloch_kx(eigenstates, ax=ax)
    ln.set_label("(23,23,12)")

    path = get_data_path("eigenstates_25_25_16_large.npy")
    eigenstates = load_eigenstate_collection(path)
    _, _, ln = plot_lowest_band_energies_against_bloch_kx(eigenstates, ax=ax)
    ln.set_label("(25,25,16)")

    ax.legend()
    ax.set_title(
        "Plot of lowest band energies\n"
        "showing convergence for an eigenstate grid of (15,15,12)"
    )

    fig.show()
    save_figure(fig, "lowest_band_convergence.png")

    fig, ax = plt.subplots()

    path = get_data_path("eigenstates_25_25_16.npy")
    eigenstates = load_eigenstate_collection(path)
    _, _, ln = plot_energies_against_bloch_phase_1d(
        eigenstates, np.array([1, 0, 0]), band=0, ax=ax
    )
    ln.set_label("n=0")
    _, _, ln = plot_energies_against_bloch_phase_1d(
        eigenstates, np.array([1, 0, 0]), band=1, ax=ax
    )
    ln.set_label("n=1")

    ax.legend()
    fig.show()
    save_figure(fig, "two_lowest_bands.png")

    fig, ax = plt.subplots()

    path = get_data_path("eigenstates_23_23_12.npy")
    eigenstates = load_eigenstate_collection(path)
    _, _, ln = plot_energies_against_bloch_phase_1d(
        eigenstates, np.array([1, 0, 0]), band=1, ax=ax
    )
    ln.set_label("(23,23,12)")

    path = get_data_path("eigenstates_25_25_16.npy")
    eigenstates = load_eigenstate_collection(path)
    _, _, ln = plot_energies_against_bloch_phase_1d(
        eigenstates, np.array([1, 0, 0]), band=1, ax=ax
    )
    ln.set_label("(25,25,16)")

    ax.legend()
    fig.show()
    save_figure(fig, "second_band_convergence.png")

    input()


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
