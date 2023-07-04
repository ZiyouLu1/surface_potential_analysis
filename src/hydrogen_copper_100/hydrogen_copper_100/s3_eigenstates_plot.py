from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from surface_potential_analysis.basis.util import Basis3dUtil
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
    plot_eigenstate_x2x0,
    plot_state_vector_along_path,
    plot_state_vector_difference_2d_x,
)

from .s3_eigenstates import get_eigenstate_collection, get_eigenstate_collection_relaxed
from .surface_data import get_data_path, save_figure

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from surface_potential_analysis.state_vector.state_vector import StateVector3d
    from surface_potential_analysis.util.util import Measure


def plot_lowest_bands() -> None:
    fig, ax = plt.subplots()

    shape = (23, 23, 14)
    for band in range(5):
        collection = get_eigenstate_collection(shape)
        _, _, ln = plot_energies_against_bloch_phase_1d(
            collection, np.array([1, 0, 0]), band=band, ax=ax
        )
        ln.set_label(f"n={band}")

    ax.legend()
    fig.show()
    input()


def plot_lowest_bands_relaxed() -> None:
    fig, ax = plt.subplots()

    shape = (23, 23, 14)
    for band in range(5):
        collection = get_eigenstate_collection_relaxed(shape)
        _, _, ln = plot_energies_against_bloch_phase_1d(
            collection, np.array([1, 0, 0]), band=band, ax=ax
        )
        ln.set_label(f"n={band}")

    ax.legend(loc="bottom right")
    fig.show()
    input()


def plot_lowest_band_energy() -> None:
    fig, ax = plt.subplots()

    shapes = [
        (23, 23, 10),
        (23, 23, 12),
        (23, 23, 14),
        (21, 21, 14),
        (21, 21, 16),
        (21, 21, 18),
    ]
    for shape in shapes:
        collection = get_eigenstate_collection(shape)
        _, _, ln = plot_energies_against_bloch_phase_1d(
            collection, np.array([1, 0, 0]), band=0, ax=ax
        )
        ln.set_label(f"({shape[0]}, {shape[1]}, {shape[2]})")

    ax.legend()
    fig.show()
    input()


def analyze_eigenvalue_convergence() -> None:
    fig, ax = plt.subplots()

    path = get_data_path("eigenstates_25_25_14.json")
    eigenstates = load_eigenstate_collection(path)
    _, _, ln = plot_lowest_band_energies_against_bloch_k(eigenstates, ax=ax)
    ln.set_label("(25,25,14)")

    path = get_data_path("eigenstates_23_23_14.json")
    eigenstates = load_eigenstate_collection(path)
    _, _, ln = plot_lowest_band_energies_against_bloch_k(eigenstates, ax=ax)
    ln.set_label("(23,23,14)")

    path = get_data_path("eigenstates_23_23_15.json")
    eigenstates = load_eigenstate_collection(path)
    _, _, ln = plot_lowest_band_energies_against_bloch_k(eigenstates, ax=ax)
    ln.set_label("(23,23,15)")

    path = get_data_path("eigenstates_23_23_16.json")
    eigenstates = load_eigenstate_collection(path)
    _, _, ln = plot_lowest_band_energies_against_bloch_k(eigenstates, ax=ax)
    ln.set_label("(23,23,16)")

    path = get_data_path("eigenstates_25_25_16.json")
    eigenstates = load_eigenstate_collection(path)
    _, _, ln = plot_lowest_band_energies_against_bloch_k(eigenstates, ax=ax)
    ln.set_label("(25,25,16)")

    path = get_data_path("eigenstates_23_23_17.json")
    eigenstates = load_eigenstate_collection(path)
    _, _, ln = plot_lowest_band_energies_against_bloch_k(eigenstates, ax=ax)
    ln.set_label("(23,23,17)")

    path = get_data_path("eigenstates_23_23_18.json")
    eigenstates = load_eigenstate_collection(path)
    _, _, ln = plot_lowest_band_energies_against_bloch_k(eigenstates, ax=ax)
    ln.set_label("(23,23,18)")

    ax.set_title(
        "Plot of energy against k for the lowest band of Copper for $K_y=0$\n"
        "showing convergence to about 2x$10^{-30}$J "
    )
    ax.set_xlabel("K /$m^{-1}$")
    ax.set_ylabel("energy / J")
    ax.legend()

    fig.tight_layout()
    fig.show()
    save_figure(fig, "copper_lowest_band_convergence.png")
    input()


def analyze_eigenvalue_convergence_relaxed() -> None:
    fig, ax = plt.subplots()

    path = get_data_path("eigenstates_relaxed_17_17_15.json")
    eigenstates = load_eigenstate_collection(path)
    _, _, ln = plot_lowest_band_energies_against_bloch_k(eigenstates, ax=ax)
    ln.set_label("(17,17,15)")

    path = get_data_path("eigenstates_relaxed_21_21_14.json")
    eigenstates = load_eigenstate_collection(path)
    _, _, ln = plot_lowest_band_energies_against_bloch_k(eigenstates, ax=ax)
    ln.set_label("(21,21,14)")

    path = get_data_path("eigenstates_relaxed_21_21_15.json")
    eigenstates = load_eigenstate_collection(path)
    _, _, ln = plot_lowest_band_energies_against_bloch_k(eigenstates, ax=ax)
    ln.set_label("(21,21,15)")

    path = get_data_path("eigenstates_relaxed_17_17_13.json")
    eigenstates = load_eigenstate_collection(path)
    _, _, ln = plot_lowest_band_energies_against_bloch_k(eigenstates, ax=ax)
    ln.set_label("(17,17,13)")

    ax.set_title(
        "Plot of energy against kx for the lowest band of Copper for $K_y=0$\n"
        "showing convergence to about 2x$10^{-30}$J "
    )
    ax.set_xlabel("K /$m^{-1}$")
    ax.set_ylabel("energy / J")
    ax.legend()

    fig.tight_layout()
    fig.show()
    save_figure(fig, "lowest_band_convergence.png")

    fig, ax = plt.subplots()

    path = get_data_path("eigenstates_relaxed_10_10_14.json")
    eigenstates = load_eigenstate_collection(path)
    _, _, ln = plot_energies_against_bloch_phase_1d(
        eigenstates, np.array([1.0, 0, 0]), 4, ax=ax
    )
    ln.set_label("(10,10,14)")

    path = get_data_path("eigenstates_relaxed_12_12_15.json")
    eigenstates = load_eigenstate_collection(path)
    _, _, ln = plot_energies_against_bloch_phase_1d(
        eigenstates, np.array([1.0, 0, 0]), 4, ax=ax
    )
    ln.set_label("(12,12,15)")

    ax.set_title(
        "Plot of energy against kx for the lowest band of Copper for $K_y=0$\n"
        "showing convergence to about 2x$10^{-30}$J "
    )
    ax.set_xlabel("K /$m^{-1}$")
    ax.set_ylabel("energy / J")
    ax.legend()

    fig.tight_layout()
    fig.show()
    save_figure(fig, "second_band_convergence.png")
    input()


def plot_lowest_eigenstate_3d_xy() -> None:
    path = get_data_path("eigenstates_relaxed_10_10_14.json")
    collection = load_eigenstate_collection(path)

    eigenstate = select_eigenstate(collection, 0, 0)

    fig, _, _anim = animate_eigenstate_x0x1(eigenstate, measure="real")
    fig.show()
    input()


def plot_eigenstate_z_hollow_site(
    eigenstate: StateVector3d[Any],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
) -> tuple[Figure, Axes, Line2D]:
    util = Basis3dUtil(eigenstate["basis"])
    x2_points = np.arange(util.n2)
    points = np.array([(util.n0 // 2, util.n1 // 2, z) for z in x2_points]).T

    return plot_state_vector_along_path(eigenstate, points, ax=ax, measure=measure)


def analyze_eigenvector_convergence_z() -> None:
    fig, ax = plt.subplots()

    path = get_data_path("eigenstates_25_25_16.json")
    collection = load_eigenstate_collection(path)
    eigenstate = select_eigenstate(collection, 0, 0)
    _, _, ln1 = plot_eigenstate_z_hollow_site(eigenstate, ax=ax)
    ln1.set_label("(25,25,16) kx=G/2")

    path = get_data_path("eigenstates_23_23_16.json")
    collection = load_eigenstate_collection(path)
    eigenstate = select_eigenstate(collection, 0, 0)
    _, _, ln2 = plot_eigenstate_z_hollow_site(eigenstate, ax=ax)
    ln2.set_label("(23,23,16) kx=G/2")

    ax.legend()
    fig.show()
    save_figure(fig, "copper_wfn_convergence.png")
    input()


def plot_eigenstate_through_bridge(
    eigenstate: StateVector3d[Any],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
) -> tuple[Figure, Axes, Line2D]:
    util = Basis3dUtil(eigenstate["basis"])
    x0_points = np.arange(util.n0)
    points = np.array([(x, util.n1 // 2, 0) for x in x0_points]).T

    return plot_state_vector_along_path(eigenstate, points, ax=ax, measure=measure)


def analyze_eigenvector_convergence_through_bridge() -> None:
    path = get_data_path("eigenstates_25_25_14.json")
    load_eigenstate_collection(path)

    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    path = get_data_path("eigenstates_23_23_15.json")
    collection = load_eigenstate_collection(path)
    eigenstate = select_eigenstate(collection, 0, 0)
    _, _, ln = plot_eigenstate_through_bridge(eigenstate, ax=ax)
    _, _, _ = plot_eigenstate_through_bridge(eigenstate, ax=ax2, measure="angle")
    ln.set_label("(23,23,15)")

    path = get_data_path("eigenstates_25_25_14.json")
    collection = load_eigenstate_collection(path)
    eigenstate = select_eigenstate(collection, 0, 0)
    _, _, ln = plot_eigenstate_through_bridge(eigenstate, ax=ax)
    _, _, _ = plot_eigenstate_through_bridge(eigenstate, ax=ax2, measure="angle")
    ln.set_label("(25,25,15)")

    ax.set_title(
        "Plot of energy against k for the lowest band of Copper for $K_y=0$\n"
        "showing convergence to about 2x$10^{-30}$J "
    )
    ax.legend()
    fig.show()
    save_figure(fig, "copper_wfn_convergence_through_bridge.png")
    input()


def plot_bloch_wavefunction_difference_at_boundary() -> None:
    path = get_data_path("eigenstates_23_23_16.json")
    collection = load_eigenstate_collection(path)
    eigenstate_0 = select_eigenstate(collection, 0, 0)

    fig, ax, _ = plot_eigenstate_x2x0(eigenstate_0, 0)
    fig.show()

    path = get_data_path("eigenstates_25_25_16.json")
    collection = load_eigenstate_collection(path)
    eigenstate_1 = select_eigenstate(collection, 0, 0)

    fig, ax, _ = plot_eigenstate_x2x0(eigenstate_1, 0)
    fig.show()

    fig, ax, _ = plot_state_vector_difference_2d_x(
        eigenstate_0, eigenstate_1, (0, 2), (0,), measure="abs", scale="linear"
    )
    ax.set_title("Divergence in the Abs value of the wavefunction")
    fig.show()

    fig, ax, _ = plot_state_vector_difference_2d_x(
        eigenstate_0, eigenstate_1, (0, 2), (0,), measure="real", scale="linear"
    )
    ax.set_title("Divergence in the real part of the wavefunction")
    fig.show()

    fig, ax, _ = plot_state_vector_difference_2d_x(
        eigenstate_0, eigenstate_1, (0, 2), (0,), measure="imag", scale="linear"
    )
    ax.set_title("Divergence in the imaginary part of the wavefunction")
    fig.show()
    input()
