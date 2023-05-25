from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt
from surface_potential_analysis.basis_config.util import BasisConfigUtil
from surface_potential_analysis.overlap.conversion import (
    convert_overlap_to_momentum_basis,
)
from surface_potential_analysis.overlap.overlap import (
    Overlap,
    load_overlap,
)
from surface_potential_analysis.overlap.plot import (
    plot_overlap_2d_k,
    plot_overlap_along_k_diagonal,
)

from .surface_data import get_data_path, save_figure

if TYPE_CHECKING:
    from surface_potential_analysis.basis_config.basis_config import (
        FundamentalPositionBasisConfig,
    )


def load_overlap_fcc_hcp() -> Overlap[FundamentalPositionBasisConfig[int, int, int]]:
    path = get_data_path("overlap_hcp_fcc.npy")
    return load_overlap(path)


def plot_overlap() -> None:
    overlap = load_overlap_fcc_hcp()
    overlap_momentum = convert_overlap_to_momentum_basis(overlap)

    fig, ax, _ = plot_overlap_2d_k(overlap_momentum, 0, 2)
    ax.set_title(
        "Plot of the overlap in momentum for ikz=0\n"
        "showing oscillation in the direction corresponding to\n"
        "a vector spanning the fcc and hcp sites"
    )
    save_figure(fig, "2d_overlap_transform_kx_ky.png")
    fig.show()

    fig, ax, _ = plot_overlap_2d_k(overlap_momentum, 0, 2)
    ax.set_title(
        "Plot of the overlap summed over z\n"
        "showing the FCC and HCP asymmetry\n"
        "in a small region in the center of the figure"
    )
    save_figure(fig, "2d_overlap_kx_ky.png")
    fig.show()

    fig, ax, _ = plot_overlap_2d_k(overlap_momentum, 0, 1)
    ax.set_title(
        "Plot of the overlap in momentum  for ikx1=0\n"
        "A very sharp peak in the kz direction"
    )
    save_figure(fig, "2d_overlap_fraction_kx1_kz.png")
    fig.show()

    fig, ax = plt.subplots()
    _, _, ln = plot_overlap_along_k_diagonal(overlap_momentum, measure="abs", ax=ax)
    ln.set_label("abs")
    _, _, ln = plot_overlap_along_k_diagonal(overlap_momentum, measure="real", ax=ax)
    ln.set_label("real")
    _, _, ln = plot_overlap_along_k_diagonal(overlap_momentum, measure="imag", ax=ax)
    ln.set_label("imag")

    ax.legend()
    ax.set_title(
        "Plot of the wavefunction along the diagonal,\nshowing an oscillation in the overlap"
    )
    save_figure(fig, "diagonal_1d_overlap_fraction.png")
    fig.show()
    input()


def print_max_overlap_momentum() -> None:
    overlap = load_overlap_fcc_hcp()
    overlap_momentum = convert_overlap_to_momentum_basis(overlap)
    util = BasisConfigUtil(overlap["basis"])

    print(overlap_momentum["vector"][0])  # noqa: T201
    print(  # noqa: T201
        np.max(np.abs(overlap_momentum["vector"].reshape(*util.shape)[:, :, 0]))
    )
    print(np.max(np.abs(overlap_momentum["vector"])))  # noqa: T201
