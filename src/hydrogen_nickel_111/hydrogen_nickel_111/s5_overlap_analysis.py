from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import Boltzmann
from surface_potential_analysis.dynamics.hermitian_gamma_integral import (
    calculate_hermitian_gamma_occupation_integral,
)
from surface_potential_analysis.overlap.conversion import (
    convert_overlap_to_momentum_basis,
)
from surface_potential_analysis.overlap.plot import (
    plot_overlap_2d_k,
    plot_overlap_2d_x,
    plot_overlap_along_k_diagonal,
)
from surface_potential_analysis.util.constants import FERMI_WAVEVECTOR

from .s5_overlap import get_overlap_hydrogen
from .surface_data import save_figure


def plot_overlap_hydrogen() -> None:
    overlap = get_overlap_hydrogen(0, 1, (0, 0), (0, -1))

    fig, ax, _ = plot_overlap_2d_x(
        overlap, (0, 1), (31,), measure="abs", scale="symlog"
    )
    ax.set_title(
        "Plot of the overlap summed over z\n"
        "showing the FCC and HCP asymmetry\n"
        "in a small region in the center of the figure"
    )
    save_figure(fig, "2d_overlap_kx_ky.png")
    fig.show()

    fig, ax, _ = plot_overlap_2d_x(overlap, (0, 1), (31,), measure="real")
    ax.set_title(
        "Plot of the overlap summed over z\n"
        "showing the FCC and HCP asymmetry\n"
        "in a small region in the center of the figure"
    )
    save_figure(fig, "2d_overlap_real_kx_ky.png")
    fig.show()
    input()


def plot_offset_overlap_hydrogen() -> None:
    overlap = get_overlap_hydrogen(0, 2, (1, 0))

    fig, _, _ = plot_overlap_2d_x(overlap, (0, 1), measure="abs")
    fig.show()

    overlap = get_overlap_hydrogen(0, 1, (0, 0), (-1, 0))

    fig, _, _ = plot_overlap_2d_x(overlap, (0, 1), measure="abs")
    fig.show()
    input()


def plot_overlap_momentum_hydrogen() -> None:
    overlap = get_overlap_hydrogen(0, 1)
    overlap_momentum = convert_overlap_to_momentum_basis(overlap)

    fig, ax, _ = plot_overlap_2d_k(overlap_momentum, (1, 0), (0,), measure="abs")
    ax.set_title(
        "Plot of the overlap in momentum for ikz=0\n"
        "showing oscillation in the direction corresponding to\n"
        "a vector spanning the fcc and hcp sites"
    )
    save_figure(fig, "2d_overlap_transform_kx_ky.png")
    fig.show()

    fig, ax, _ = plot_overlap_2d_k(overlap_momentum, (1, 0), (0,), measure="real")
    ax.set_title(
        "Plot of the overlap in momentum for ikz=0\n"
        "showing oscillation in the direction corresponding to\n"
        "a vector spanning the fcc and hcp sites"
    )
    save_figure(fig, "2d_overlap_transform_real_kx_ky.png")
    fig.show()

    fig, ax, _ = plot_overlap_2d_k(overlap_momentum, (1, 0), (0,), measure="imag")
    ax.set_title(
        "Plot of the overlap in momentum for ikz=0\n"
        "showing oscillation in the direction corresponding to\n"
        "a vector spanning the fcc and hcp sites"
    )
    save_figure(fig, "2d_overlap_transform_imag_kx_ky.png")
    fig.show()

    fig, ax, _ = plot_overlap_2d_k(overlap_momentum, (0, 2))
    ax.set_title(
        "Plot of the overlap in momentum for ikx1=0\n"
        "A very sharp peak in the kz direction"
    )
    ax.set_ylim(-4e11, 4e11)
    save_figure(fig, "2d_overlap_fraction_kx1_kz.png")
    fig.show()
    input()


def plot_overlap_momentum_along_diagonal() -> None:
    overlap = get_overlap_hydrogen(0, 1)
    overlap_momentum = convert_overlap_to_momentum_basis(overlap)

    fig, ax = plt.subplots()
    _, _, ln = plot_overlap_along_k_diagonal(overlap_momentum, 2, measure="abs", ax=ax)
    ln.set_label("abs")
    _, _, ln = plot_overlap_along_k_diagonal(overlap_momentum, 2, measure="real", ax=ax)
    ln.set_label("real")
    _, _, ln = plot_overlap_along_k_diagonal(overlap_momentum, 2, measure="imag", ax=ax)
    ln.set_label("imag")

    ax.legend()
    ax.set_title(
        "Plot of the wavefunction along the diagonal,\nshowing an oscillation in the overlap"
    )

    save_figure(fig, "diagonal_1d_overlap_fraction.png")
    fig.show()
    input()


def plot_temperature_dependent_integral() -> None:
    temperatures = np.linspace(50, 200, 50)
    vals = [
        calculate_hermitian_gamma_occupation_integral(
            0, FERMI_WAVEVECTOR["NICKEL"], Boltzmann * t
        )
        for t in temperatures
    ]
    fig, ax = plt.subplots()
    ax.plot(1 / temperatures, vals)

    fig.show()
    input()
