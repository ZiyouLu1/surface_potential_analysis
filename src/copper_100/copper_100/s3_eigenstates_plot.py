from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
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

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D


def analyze_eigenvalue_convergence() -> None:
    fig, ax = plt.subplots()

    path = get_data_path("eigenstates_25_25_14.json")
    eigenstates = load_eigenstate_collection(path)
    _, _, ln = plot_lowest_band_energies_against_bloch_kx(eigenstates, ax=ax)
    ln.set_label("(25,25,14)")

    path = get_data_path("eigenstates_23_23_14.json")
    eigenstates = load_eigenstate_collection(path)
    _, _, ln = plot_lowest_band_energies_against_bloch_kx(eigenstates, ax=ax)
    ln.set_label("(23,23,14)")

    path = get_data_path("eigenstates_23_23_15.json")
    eigenstates = load_eigenstate_collection(path)
    _, _, ln = plot_lowest_band_energies_against_bloch_kx(eigenstates, ax=ax)
    ln.set_label("(23,23,15)")

    path = get_data_path("eigenstates_23_23_16.json")
    eigenstates = load_eigenstate_collection(path)
    _, _, ln = plot_lowest_band_energies_against_bloch_kx(eigenstates, ax=ax)
    ln.set_label("(23,23,16)")

    path = get_data_path("eigenstates_25_25_16.json")
    eigenstates = load_eigenstate_collection(path)
    _, _, ln = plot_lowest_band_energies_against_bloch_kx(eigenstates, ax=ax)
    ln.set_label("(25,25,16)")

    path = get_data_path("eigenstates_23_23_17.json")
    eigenstates = load_eigenstate_collection(path)
    _, _, ln = plot_lowest_band_energies_against_bloch_kx(eigenstates, ax=ax)
    ln.set_label("(23,23,17)")

    path = get_data_path("eigenstates_23_23_18.json")
    eigenstates = load_eigenstate_collection(path)
    _, _, ln = plot_lowest_band_energies_against_bloch_kx(eigenstates, ax=ax)
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
    _, _, ln = plot_lowest_band_energies_against_bloch_kx(eigenstates, ax=ax)
    ln.set_label("(17,17,15)")

    path = get_data_path("eigenstates_relaxed_21_21_14.json")
    eigenstates = load_eigenstate_collection(path)
    _, _, ln = plot_lowest_band_energies_against_bloch_kx(eigenstates, ax=ax)
    ln.set_label("(21,21,14)")

    path = get_data_path("eigenstates_relaxed_21_21_15.json")
    eigenstates = load_eigenstate_collection(path)
    _, _, ln = plot_lowest_band_energies_against_bloch_kx(eigenstates, ax=ax)
    ln.set_label("(21,21,15)")

    path = get_data_path("eigenstates_relaxed_17_17_13.json")
    eigenstates = load_eigenstate_collection(path)
    _, _, ln = plot_lowest_band_energies_against_bloch_kx(eigenstates, ax=ax)
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
    eigenstate: Eigenstate,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    util = EigenstateConfigUtil(config)
    z_points = np.linspace(-util.characteristic_z * 2, util.characteristic_z * 2, 1000)
    points = np.array(
        [(util.delta_x0[0] / 2, util.delta_x1[1] / 2, z) for z in z_points]
    )

    return plot_eigenstate_along_path(config, eigenstate, points, ax=ax)


def analyze_eigenvector_convergence_z() -> None:
    fig, ax = plt.subplots()

    path = get_data_path("eigenstates_25_25_16.json")
    eigenstates = load_eigenstate_collection(path)
    _, _, ln = plot_eigenstate_z_hollow_site(
        eigenstates["eigenstate_config"], get_eigenstate_list(eigenstates)[0], ax=ax
    )
    ln.set_label("(25,25,16) kx=G/2")

    path = get_data_path("eigenstates_23_23_16.json")
    eigenstates = load_eigenstate_collection(path)
    _, _, l2 = plot_eigenstate_z_hollow_site(
        eigenstates["eigenstate_config"], get_eigenstate_list(eigenstates)[0], ax=ax
    )
    l2.set_label("(23,23,16) kx=G/2")

    ax.legend()
    fig.show()
    save_figure(fig, "copper_wfn_convergence.png")
    input()


def plot_eigenstate_through_bridge(
    config: EigenstateConfig,
    eigenstate: Eigenstate,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs", "angle"] = "abs",
) -> tuple[Figure, Axes, Line2D]:
    util = EigenstateConfigUtil(config)

    x_points = np.linspace(0, util.delta_x0[0], 1000)
    points = np.array([(x, util.delta_x1[1] / 2, 0) for x in x_points])
    return plot_eigenstate_along_path(
        config, eigenstate, points, ax=ax, measure=measure
    )


def analyze_eigenvector_convergence_through_bridge() -> None:
    path = get_data_path("eigenstates_25_25_14.json")
    eigenstates = load_eigenstate_collection(path)

    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    path = get_data_path("eigenstates_23_23_15.json")
    eigenstates = load_eigenstate_collection(path)
    normalized = normalize_eigenstate_phase(eigenstates)
    _, _, ln = plot_eigenstate_through_bridge(
        eigenstates["eigenstate_config"], get_eigenstate_list(normalized)[5], ax=ax
    )
    _, _, _ = plot_eigenstate_through_bridge(
        eigenstates["eigenstate_config"],
        get_eigenstate_list(eigenstates)[5],
        ax=ax2,
        measure="angle",
    )
    ln.set_label("(23,23,15)")

    path = get_data_path("eigenstates_25_25_14.json")
    eigenstates = load_energy_eigenstates(path)
    normalized = normalize_eigenstate_phase(eigenstates)
    _, _, ln = plot_eigenstate_through_bridge(
        eigenstates["eigenstate_config"], get_eigenstate_list(normalized)[5], ax=ax
    )
    _, _, _ = plot_eigenstate_through_bridge(
        eigenstates["eigenstate_config"],
        get_eigenstate_list(eigenstates)[5],
        ax=ax2,
        measure="angle",
    )
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
    eigenstates0 = load_eigenstate_collection(path)
    eigenstate0 = get_eigenstate_list(eigenstates0)[0]

    fig, ax, _ = plot_eigenstate_x0z(eigenstates0["eigenstate_config"], eigenstate0)
    fig.show()

    path = get_data_path("eigenstates_25_25_16.json")
    eigenstates1 = load_eigenstate_collection(path)
    eigenstate1 = get_eigenstate_list(eigenstates1)[0]

    fig, ax, _ = plot_eigenstate_x0z(eigenstates1["eigenstate_config"], eigenstate1)
    fig.show()

    fig, ax, _ = plot_bloch_wavefunction_difference_in_x0z(
        eigenstates0["eigenstate_config"],
        eigenstate0,
        eigenstates1["eigenstate_config"],
        eigenstate1,
        measure="abs",
        norm="linear",
    )
    ax.set_title("Divergence in the Abs value of the wavefunction")
    fig.show()

    fig, ax, _ = plot_bloch_wavefunction_difference_in_x0z(
        eigenstates0["eigenstate_config"],
        eigenstate0,
        eigenstates1["eigenstate_config"],
        eigenstate1,
        measure="real",
        norm="linear",
    )
    ax.set_title("Divergence in the real part of the wavefunction")
    fig.show()

    fig, ax, _ = plot_bloch_wavefunction_difference_in_x0z(
        eigenstates0["eigenstate_config"],
        eigenstate0,
        eigenstates1["eigenstate_config"],
        eigenstate1,
        measure="imag",
        norm="linear",
    )
    ax.set_title("Divergence in the imaginary part of the wavefunction")
    fig.show()
    input()


def analyze_oversampling_effect() -> None:
    """
    Check if the effect of sampling a larger grid of points in k space.

    Does this change the eigenvector of the groundstate significantly?.
    """
    path = get_data_path("oversampled_eigenstates.json")
    oversampled = load_energy_eigenstates(path)

    path = get_data_path("not_oversampled_eigenstates.json")
    not_oversampled = load_energy_eigenstates(path)

    print("oversampled groundstate", np.min(oversampled["eigenvalues"]))
    print("not oversampled groundstate", np.min(not_oversampled["eigenvalues"]))

    oversampled_gs = oversampled["eigenvectors"][np.argmin(oversampled["eigenvalues"])]
    not_oversampled_gs = not_oversampled["eigenvectors"][
        np.argmin(not_oversampled["eigenvalues"])
    ]

    util_over = EigenstateConfigUtil(oversampled["eigenstate_config"])
    util_not_over = EigenstateConfigUtil(not_oversampled["eigenstate_config"])
    product = 0
    for ix0, ix1, ix2 in util_over.eigenstate_indexes:
        product += oversampled_gs[util_over.get_index(ix0, ix1, ix2)] * np.conj(
            not_oversampled_gs[util_not_over.get_index(ix0, ix1, ix2)]
        )
    print("product", product)
