import numpy as np
from matplotlib import pyplot as plt

from surface_potential_analysis.overlap_transform import load_overlap_transform
from surface_potential_analysis.overlap_transform_plot import (
    plot_overlap_transform_along_diagonal,
    plot_overlap_transform_x0z,
    plot_overlap_transform_xy,
    plot_overlap_xy,
)

from .surface_data import get_data_path, save_figure


def plot_overlap():
    path = get_data_path("overlap_transform_hcp_fcc.npz")
    # path = get_data_path("overlap_transform_interpolated_hcp_fcc.npz")
    overlap = load_overlap_transform(path)

    fig, ax, _ = plot_overlap_transform_xy(overlap)
    ax.set_title(
        "Plot of the overlap transform for ikz=0\n"
        "showing oscillation in the direction corresponding to\n"
        "a vector spanning the fcc and hcp sites"
    )
    save_figure(fig, "2d_overlap_transform_kx_ky.png")
    fig.show()

    fig, ax, _ = plot_overlap_xy(overlap)
    ax.set_title(
        "Plot of the overlap summed over z\n"
        "showing the FCC and HCP asymmetry\n"
        "in a small region in the center of the figure"
    )
    save_figure(fig, "2d_overlap_kx_ky.png")
    fig.show()

    fig, ax, _ = plot_overlap_transform_x0z(overlap)
    ax.set_title(
        "Plot of the overlap transform for ikx1=0\n"
        "A very sharp peak in the kz direction"
    )
    save_figure(fig, "2d_overlap_fraction_kx1_kz.png")
    fig.show()

    fig, ax = plt.subplots()
    _, _, ln = plot_overlap_transform_along_diagonal(overlap, measure="abs", ax=ax)
    ln.set_label("abs")
    _, _, ln = plot_overlap_transform_along_diagonal(overlap, measure="real", ax=ax)
    ln.set_label("real")
    _, _, ln = plot_overlap_transform_along_diagonal(overlap, measure="imag", ax=ax)
    ln.set_label("imag")

    # ax2 = ax.twinx()
    # _, _, ln = plot_overlap_transform_along_diagonal(overlap, measure="angle", ax=ax2)
    # ln.set_label("angle")

    ax.legend()
    ax.set_title(
        "Plot of the wavefunction along the diagonal,\nshowing an oscillation in the overlap"
    )
    save_figure(fig, "diagonal_1d_overlap_fraction.png")
    fig.show()
    input()


def fit_overlap_transform():
    path = get_data_path("overlap_transform_hcp_fcc.npz")
    overlap = load_overlap_transform(path)
    points = overlap["points"]

    print(points[0, 0, 0])
    print(np.max(np.abs(points[:, :, 0])))
    print(np.max(np.abs(points[:, :])))
