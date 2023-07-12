from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from surface_potential_analysis.overlap.conversion import (
    convert_overlap_to_momentum_basis,
)
from surface_potential_analysis.overlap.overlap import (
    load_overlap,
)
from surface_potential_analysis.overlap.plot import (
    plot_overlap_2d_k,
    plot_overlap_2d_x,
    plot_overlap_along_k0,
)

from .surface_data import get_data_path, save_figure


def plot_overlap() -> None:
    path = get_data_path("overlap_transform_0_next_0.npz")
    path = get_data_path("overlap_transform_large_0_next_0.npz")
    overlap = load_overlap(path)
    overlap_momentum = convert_overlap_to_momentum_basis(overlap)

    fig, ax, _ = plot_overlap_2d_k(overlap_momentum, (0, 1), (0,))
    ax.set_title(
        "Plot of the overlap in momentum for ikz=0\n"
        "showing oscillation in the direction corresponding to\n"
        "a vector spanning the two sites"
    )
    save_figure(fig, "2d_overlap_transform_kx_ky.png")
    fig.show()

    fig, ax, _ = plot_overlap_2d_k(overlap_momentum, (0, 1), (0,), measure="real")
    ax.set_title(
        "Plot of the overlap in momentum for ikz=0\n"
        "showing oscillation in the direction corresponding to\n"
        "a vector spanning the two sites"
    )
    save_figure(fig, "2d_overlap_transform_real_kx_ky.png")
    fig.show()

    fig, ax, _ = plot_overlap_2d_k(overlap_momentum, (0, 1), (0,), measure="imag")
    ax.set_title(
        "Plot of the overlap in momentum for ikz=0\n"
        "showing oscillation in the direction corresponding to\n"
        "a vector spanning the two sites"
    )
    save_figure(fig, "2d_overlap_transform_imag_kx_ky.png")
    fig.show()

    fig, ax, _ = plot_overlap_2d_x(overlap, (0, 1), (0,))
    ax.set_title("Plot of the overlap summed over z")
    save_figure(fig, "2d_overlap_kx_ky.png")
    fig.show()

    fig, ax, _ = plot_overlap_2d_k(overlap_momentum, (2, 0), (0,))
    ax.set_title(
        "Plot of the overlap in momentum for ikx1=0 with a decay in the kz direction"
    )
    ax.set_ylim(0, 2e11)
    save_figure(fig, "2d_overlap_fraction_kx1_kz.png")
    fig.show()

    fig, ax = plt.subplots()
    _, _, ln = plot_overlap_along_k0(overlap_momentum, measure="abs", ax=ax)
    ln.set_label("abs")
    _, _, ln = plot_overlap_along_k0(overlap_momentum, measure="real", ax=ax)
    ln.set_label("real")
    _, _, ln = plot_overlap_along_k0(overlap_momentum, measure="imag", ax=ax)
    ln.set_label("imag")

    ax.legend()
    ax.set_title(
        "Plot of the wavefunction along the diagonal,\nshowing an oscillation in the overlap"
    )
    save_figure(fig, "diagonal_1d_overlap_fraction.png")
    fig.show()
    input()


def fit_overlap_momentum() -> None:
    path = get_data_path("overlap_transform_0_next_0.npz")
    overlap = load_overlap(path)
    overlap_momentum = convert_overlap_to_momentum_basis(overlap)
    points = overlap_momentum["vector"]

    print(points[0, 0, 0])  # noqa: T201
    print(points.shape)  # noqa: T201
    print(np.max(np.abs(points[:, :, 0])))  # noqa: T201
    print(np.max(np.abs(points[:, :])))  # noqa: T201

    path = get_data_path("overlap_transform_large_0_next_0.npz")
    overlap = load_overlap(path)
    overlap_momentum = convert_overlap_to_momentum_basis(overlap)
    points = overlap_momentum["vector"]

    print(points[0, 0, 0])  # noqa: T201
    print(points.shape)  # noqa: T201
    print(np.max(np.abs(points[:, :, 0])))  # noqa: T201
    print(np.max(np.abs(points[:, :])))  # noqa: T201
