from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from surface_potential_analysis.state_vector.eigenstate_collection import (
    load_eigenstate_collection,
    select_eigenstate,
)
from surface_potential_analysis.state_vector.eigenstate_collection_plot import (
    plot_eigenvalues_against_bloch_phase_1d,
    plot_lowest_band_eigenvalues_against_bloch_k,
)
from surface_potential_analysis.state_vector.plot import animate_eigenstate_x0x1

from .surface_data import get_data_path, save_figure


def analyze_band_convergence() -> None:
    fig, ax = plt.subplots()

    path = get_data_path("eigenstates_23_23_10_large.npy")
    eigenstates = load_eigenstate_collection(path)
    _, _, ln = plot_lowest_band_eigenvalues_against_bloch_k(eigenstates, ax=ax)
    ln.set_label("(23,23,10)")

    path = get_data_path("eigenstates_23_23_12.npy")
    eigenstates = load_eigenstate_collection(path)
    _, _, ln = plot_lowest_band_eigenvalues_against_bloch_k(eigenstates, ax=ax)
    ln.set_label("(23,23,12)")

    path = get_data_path("eigenstates_25_25_16.npy")
    eigenstates = load_eigenstate_collection(path)
    _, _, ln = plot_lowest_band_eigenvalues_against_bloch_k(eigenstates, ax=ax)
    ln.set_label("(25,25,16)")

    ax.legend()
    ax.set_title(
        "Plot of lowest band energies\n"
        "showing convergence for an eigenstate grid of (23,23,16)"
    )

    fig.show()
    save_figure(fig, "lowest_band_convergence.png")

    fig, ax = plt.subplots()

    path = get_data_path("eigenstates_25_25_16.npy")
    eigenstates = load_eigenstate_collection(path)
    _, _, ln = plot_eigenvalues_against_bloch_phase_1d(
        eigenstates, np.array([1, 0, 0]), band=0, ax=ax
    )
    ln.set_label("n=0")
    _, _, ln = plot_eigenvalues_against_bloch_phase_1d(
        eigenstates, np.array([1, 0, 0]), band=1, ax=ax
    )
    ln.set_label("n=1")

    ax.legend()
    fig.show()
    save_figure(fig, "two_lowest_bands.png")

    fig, ax = plt.subplots()

    path = get_data_path("eigenstates_23_23_12.npy")
    eigenstates = load_eigenstate_collection(path)
    _, _, ln = plot_eigenvalues_against_bloch_phase_1d(
        eigenstates, np.array([1, 0, 0]), band=1, ax=ax
    )
    ln.set_label("(23,23,12)")

    path = get_data_path("eigenstates_25_25_16.npy")
    eigenstates = load_eigenstate_collection(path)
    _, _, ln = plot_eigenvalues_against_bloch_phase_1d(
        eigenstates, np.array([1, 0, 0]), band=1, ax=ax
    )
    ln.set_label("(25,25,16)")

    ax.legend()
    fig.show()
    save_figure(fig, "second_band_convergence.png")

    input()


def plot_eigenstate_for_each_band() -> None:
    """Check to see if the eigenstates look as they are supposed to."""
    path = get_data_path("eigenstates_29_29_12.json")
    eigenstates = load_eigenstate_collection(path)

    eigenstate = select_eigenstate(eigenstates, bloch_idx=0, band_idx=0)
    fig, _, _anim1 = animate_eigenstate_x0x1(eigenstate)
    fig.show()

    eigenstate = select_eigenstate(eigenstates, bloch_idx=0, band_idx=1)
    fig, _, _anim2 = animate_eigenstate_x0x1(eigenstate)
    fig.show()
    input()
