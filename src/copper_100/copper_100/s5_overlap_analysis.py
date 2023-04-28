from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from surface_potential_analysis.overlap_transform import load_overlap_transform
from surface_potential_analysis.overlap_transform_plot import (
    plot_overlap_transform_along_x0,
    plot_overlap_transform_x0z,
    plot_overlap_transform_xy,
    plot_overlap_xy,
)

from .surface_data import get_data_path, save_figure


def plot_overlap():
    path = get_data_path("overlap_transform_0_next_0.npz")
    path = get_data_path("overlap_transform_large_0_next_0.npz")
    overlap = load_overlap_transform(path)

    fig, ax, _ = plot_overlap_transform_xy(overlap)
    ax.set_title(
        "Plot of the overlap transform for ikz=0\n"
        "showing oscillation in the direction corresponding to\n"
        "a vector spanning the two sites"
    )
    save_figure(fig, "2d_overlap_transform_kx_ky.png")
    fig.show()

    fig, ax, _ = plot_overlap_transform_xy(overlap, measure="real")
    ax.set_title(
        "Plot of the overlap transform for ikz=0\n"
        "showing oscillation in the direction corresponding to\n"
        "a vector spanning the two sites"
    )
    save_figure(fig, "2d_overlap_transform_real_kx_ky.png")
    fig.show()

    fig, ax, _ = plot_overlap_transform_xy(overlap, measure="imag")
    ax.set_title(
        "Plot of the overlap transform for ikz=0\n"
        "showing oscillation in the direction corresponding to\n"
        "a vector spanning the two sites"
    )
    save_figure(fig, "2d_overlap_transform_imag_kx_ky.png")
    fig.show()

    fig, ax, _ = plot_overlap_xy(overlap)
    ax.set_title("Plot of the overlap summed over z")
    save_figure(fig, "2d_overlap_kx_ky.png")
    fig.show()

    fig, ax, _ = plot_overlap_transform_x0z(overlap)
    ax.set_title(
        "Plot of the overlap transform for ikx1=0\n" "with a decay in the kz direction"
    )
    ax.set_ylim(0, 2e11)
    save_figure(fig, "2d_overlap_fraction_kx1_kz.png")
    fig.show()

    fig, ax = plt.subplots()
    _, _, ln = plot_overlap_transform_along_x0(overlap, measure="abs", ax=ax)
    ln.set_label("abs")
    _, _, ln = plot_overlap_transform_along_x0(overlap, measure="real", ax=ax)
    ln.set_label("real")
    _, _, ln = plot_overlap_transform_along_x0(overlap, measure="imag", ax=ax)
    ln.set_label("imag")


    ax.legend()
    ax.set_title(
        "Plot of the wavefunction along the diagonal,\nshowing an oscillation in the overlap"
    )
    save_figure(fig, "diagonal_1d_overlap_fraction.png")
    fig.show()
    input()


def fit_overlap_transform():
    path = get_data_path("overlap_transform_0_next_0.npz")
    overlap = load_overlap_transform(path)
    points = overlap["points"]

    print(points[0, 0, 0])
    print(points.shape)
    print(np.max(np.abs(points[:, :, 0])))
    print(np.max(np.abs(points[:, :])))

    path = get_data_path("overlap_transform_large_0_next_0.npz")
    overlap = load_overlap_transform(path)
    points = overlap["points"]

    print(points[0, 0, 0])
    print(points.shape)
    print(np.max(np.abs(points[:, :, 0])))
    print(np.max(np.abs(points[:, :])))
