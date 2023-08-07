from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from surface_potential_analysis.basis.util import AxisWithLengthBasisUtil
from surface_potential_analysis.state_vector.eigenstate_collection import (
    select_eigenstate,
)
from surface_potential_analysis.state_vector.eigenstate_collection_plot import (
    plot_eigenvalues_against_bloch_phase_1d,
)
from surface_potential_analysis.state_vector.plot import (
    animate_state_x0x1,
    plot_state_along_path,
    plot_state_difference_2d_x,
    plot_state_x2x0,
)

from .s3_eigenstates import get_eigenstate_collection, get_eigenstate_collection_relaxed
from .surface_data import save_figure

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
        _, _, ln = plot_eigenvalues_against_bloch_phase_1d(
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
        _, _, ln = plot_eigenvalues_against_bloch_phase_1d(
            collection, np.array([1, 0, 0]), band=band, ax=ax
        )
        ln.set_label(f"n={band}")

    ax.legend(loc="bottom right")
    fig.show()
    input()


def plot_lowest_band_energy() -> None:
    fig, ax = plt.subplots()

    shapes = [
        (21, 21, 12),
        (21, 21, 14),
        (21, 21, 15),
        (21, 21, 16),
        (23, 23, 15),
    ]
    for shape in shapes:
        collection = get_eigenstate_collection(shape)
        _, _, ln = plot_eigenvalues_against_bloch_phase_1d(
            collection, np.array([1, 0, 0]), band=0, ax=ax
        )
        ln.set_label(f"({shape[0]}, {shape[1]}, {shape[2]})")

    ax.legend()
    fig.show()
    input()


def plot_lowest_band_energy_relaxed() -> None:
    fig, ax = plt.subplots()

    shapes = [
        (17, 17, 14),
        (19, 19, 14),
        (21, 21, 14),
        (21, 21, 16),
    ]
    for shape in shapes:
        collection = get_eigenstate_collection_relaxed(shape)
        _, _, ln = plot_eigenvalues_against_bloch_phase_1d(
            collection, np.array([1, 0, 0]), band=0, ax=ax
        )
        ln.set_label(f"({shape[0]}, {shape[1]}, {shape[2]})")

    ax.legend()
    fig.show()
    input()


def plot_lowest_eigenstate_3d_xy() -> None:
    collection = get_eigenstate_collection_relaxed((10, 10, 14))
    eigenstate = select_eigenstate(collection, 0, 0)

    fig, _, _anim = animate_state_x0x1(eigenstate, measure="real")
    fig.show()
    input()


def plot_eigenstate_z_hollow_site(
    eigenstate: StateVector3d[Any],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
) -> tuple[Figure, Axes, Line2D]:
    util = AxisWithLengthBasisUtil(eigenstate["basis"])
    x2_points = np.arange(util.shape[2])
    points = np.array(
        [(util.shape[0] // 2, util.shape[1] // 2, z) for z in x2_points]
    ).T

    return plot_state_along_path(eigenstate, points, ax=ax, measure=measure)


def analyze_eigenvector_convergence_z() -> None:
    fig, ax = plt.subplots()

    collection = get_eigenstate_collection((25, 25, 16))
    eigenstate = select_eigenstate(collection, 0, 0)
    _, _, ln1 = plot_eigenstate_z_hollow_site(eigenstate, ax=ax)
    ln1.set_label("(25,25,16) kx=G/2")

    collection = get_eigenstate_collection((23, 23, 16))
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
    util = AxisWithLengthBasisUtil(eigenstate["basis"])
    x0_points = np.arange(util.shape[0])
    points = np.array([(x, util.shape[1] // 2, 0) for x in x0_points]).T

    return plot_state_along_path(eigenstate, points, ax=ax, measure=measure)


def analyze_eigenvector_convergence_through_bridge() -> None:
    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    collection = get_eigenstate_collection((23, 23, 15))
    eigenstate = select_eigenstate(collection, 0, 0)
    _, _, ln = plot_eigenstate_through_bridge(eigenstate, ax=ax)
    _, _, _ = plot_eigenstate_through_bridge(eigenstate, ax=ax2, measure="angle")
    ln.set_label("(23,23,15)")

    collection = get_eigenstate_collection((25, 25, 14))
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
    collection = get_eigenstate_collection((23, 23, 16))
    eigenstate_0 = select_eigenstate(collection, 0, 0)

    fig, ax, _ = plot_state_x2x0(eigenstate_0, 0)
    fig.show()

    collection = get_eigenstate_collection((25, 25, 16))
    eigenstate_1 = select_eigenstate(collection, 0, 0)

    fig, ax, _ = plot_state_x2x0(eigenstate_1, 0)
    fig.show()

    fig, ax, _ = plot_state_difference_2d_x(
        eigenstate_0, eigenstate_1, (0, 2), (0,), measure="abs", scale="linear"
    )
    ax.set_title("Divergence in the Abs value of the wavefunction")
    fig.show()

    fig, ax, _ = plot_state_difference_2d_x(
        eigenstate_0, eigenstate_1, (0, 2), (0,), measure="real", scale="linear"
    )
    ax.set_title("Divergence in the real part of the wavefunction")
    fig.show()

    fig, ax, _ = plot_state_difference_2d_x(
        eigenstate_0, eigenstate_1, (0, 2), (0,), measure="imag", scale="linear"
    )
    ax.set_title("Divergence in the imaginary part of the wavefunction")
    fig.show()
    input()
