from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import Boltzmann
from surface_dynamics_simulation.simulation_config import (
    build_tunnelling_matrix,
    simulate_tunnelling,
)
from surface_potential_analysis.basis_config.basis_config import (
    BasisConfigUtil,
    MomentumBasisConfig,
    PositionBasisConfig,
)
from surface_potential_analysis.overlap.overlap import (
    Overlap,
    OverlapTransform,
    convert_overlap_momentum_basis,
    load_overlap,
)
from surface_potential_analysis.overlap.plot import (
    plot_overlap_2d_k,
    plot_overlap_2d_x,
    plot_overlap_along_k_diagonal,
    plot_overlap_k0k1,
)
from surface_potential_analysis.util import npy_cached

from nickel_111.s4_wavepacket import load_nickel_wavepacket

from .surface_data import get_data_path, save_figure

if TYPE_CHECKING:
    from surface_potential_analysis._types import SingleIndexLike

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)


def load_overlap_fcc_hcp() -> Overlap[PositionBasisConfig[int, int, int]]:
    path = get_data_path("overlap_hcp_fcc.npy")
    return load_overlap(path)


def load_overlap_fcc_fcc() -> Overlap[PositionBasisConfig[int, int, int]]:
    path = get_data_path("overlap_fcc_fcc.npy")
    return load_overlap(path)


def load_overlap_hcp_hcp() -> Overlap[PositionBasisConfig[int, int, int]]:
    path = get_data_path("overlap_hcp_hcp.npy")
    return load_overlap(path)


def get_max_point(
    overlap: OverlapTransform[_L0Inv, _L1Inv, _L2Inv]
) -> tuple[int, int, int]:
    points = overlap["vector"]
    (ik0, ik1, inz) = np.unravel_index(np.argmax(np.abs(points)), shape=points.shape)
    return (int(ik0), int(ik1), int(inz))


def make_transform_real_at(
    overlap: OverlapTransform[_L0Inv, _L1Inv, _L2Inv],
    idx: SingleIndexLike | None = None,
) -> OverlapTransform[_L0Inv, _L1Inv, _L2Inv]:
    """
    Shift the phase of the overlap transform such that is it real at point.

    This is equivalent to shifting the origin of the real space grid we use when
    calculating the overlap.

    Parameters
    ----------
    overlap : OverlapTransform
    point : tuple[int, int, int] | None, optional
        Point to make real, by default the maximum of the transform

    Returns
    -------
    OverlapTransform
        A new overlap transform, which is real at the given point
    """
    util = BasisConfigUtil(overlap["basis"])
    idx = int(np.argmax(np.abs(overlap["vector"]))) if idx is None else idx
    idx = util.get_flat_index(idx) if isinstance(idx, tuple) else idx

    new_points = overlap["vector"] * np.exp(-1j * np.angle(overlap["vector"][idx]))
    return {"basis": overlap["basis"], "vector": new_points}


def plot_fcc_hcp_overlap_transform() -> None:
    overlap = load_overlap_fcc_hcp()
    overlap_transform = convert_overlap_momentum_basis(overlap)

    fig, ax, _ = plot_overlap_2d_k(overlap_transform, 0, 2, measure="abs")
    ax.set_title(
        "Plot of the overlap transform for ikz=0\n"
        "showing oscillation in the direction corresponding to\n"
        "a vector spanning the fcc and hcp sites"
    )
    save_figure(fig, "2d_overlap_transform_kx_ky.png")
    fig.show()

    fig, ax, _ = plot_overlap_2d_k(overlap_transform, 0, 2, measure="real")
    ax.set_title(
        "Plot of the overlap transform for ikz=0\n"
        "showing oscillation in the direction corresponding to\n"
        "a vector spanning the fcc and hcp sites"
    )
    save_figure(fig, "2d_overlap_transform_real_kx_ky.png")
    fig.show()

    fig, ax, _ = plot_overlap_2d_k(overlap_transform, 0, 2, measure="imag")
    ax.set_title(
        "Plot of the overlap transform for ikz=0\n"
        "showing oscillation in the direction corresponding to\n"
        "a vector spanning the fcc and hcp sites"
    )
    save_figure(fig, "2d_overlap_transform_imag_kx_ky.png")
    fig.show()

    fig, ax, _ = plot_overlap_2d_k(overlap_transform, 0, 0)
    ax.set_title(
        "Plot of the overlap transform for ikx1=0\n"
        "A very sharp peak in the kz direction"
    )
    ax.set_ylim(-4e11, 4e11)
    save_figure(fig, "2d_overlap_fraction_kx1_kz.png")
    fig.show()
    input()


def plot_fcc_hcp_overlap_transform_along_diagonal() -> None:
    overlap = load_overlap_fcc_hcp()
    overlap_transform = convert_overlap_momentum_basis(overlap)

    fig, ax = plt.subplots()
    _, _, ln = plot_overlap_along_k_diagonal(overlap_transform, 2, measure="abs", ax=ax)
    ln.set_label("abs")
    _, _, ln = plot_overlap_along_k_diagonal(
        overlap_transform, 2, measure="real", ax=ax
    )
    ln.set_label("real")
    _, _, ln = plot_overlap_along_k_diagonal(
        overlap_transform, 2, measure="imag", ax=ax
    )
    ln.set_label("imag")

    ax.legend()
    ax.set_title(
        "Plot of the wavefunction along the diagonal,\nshowing an oscillation in the overlap"
    )

    save_figure(fig, "diagonal_1d_overlap_fraction.png")
    fig.show()
    input()


def plot_fcc_hcp_overlap() -> None:
    overlap = load_overlap_fcc_hcp()
    fig, ax, _ = plot_overlap_2d_x(overlap, 177, 2, measure="abs")
    ax.set_title(
        "Plot of the overlap summed over z\n"
        "showing the FCC and HCP asymmetry\n"
        "in a small region in the center of the figure"
    )
    save_figure(fig, "2d_overlap_kx_ky.png")
    fig.show()

    fig, ax, _ = plot_overlap_2d_x(overlap, 177, 2, measure="real")
    ax.set_title(
        "Plot of the overlap summed over z\n"
        "showing the FCC and HCP asymmetry\n"
        "in a small region in the center of the figure"
    )
    save_figure(fig, "2d_overlap_real_kx_ky.png")
    fig.show()
    input()


def calculate_max_overlap(
    overlap: Overlap[PositionBasisConfig[_L0Inv, _L1Inv, _L2Inv]],
) -> tuple[complex, np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]]:
    points = overlap["vector"]
    util = BasisConfigUtil(overlap["basis"])
    arg_max = np.argmax(np.abs(points))
    x_point = util.fundamental_x_points[:, arg_max]

    return points[arg_max], x_point


def calculate_max_overlap_transform(
    overlap: Overlap[MomentumBasisConfig[_L0Inv, _L1Inv, _L2Inv]],
) -> tuple[complex, np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]]:
    points = overlap["vector"]
    util = BasisConfigUtil(overlap["basis"])
    arg_max = np.argmax(np.abs(points))
    k_point = util.fundamental_k_points[:, arg_max]

    return points[arg_max], k_point


def print_max_overlap_transforms() -> None:
    overlap = load_overlap_fcc_hcp()
    overlap_transform = convert_overlap_momentum_basis(overlap)
    print(calculate_max_overlap_transform(overlap_transform))  # noqa: T201

    overlap = load_overlap_fcc_fcc()
    overlap_transform = convert_overlap_momentum_basis(overlap)
    print(calculate_max_overlap_transform(overlap_transform))  # noqa: T201

    overlap = load_overlap_hcp_hcp()
    overlap_transform = convert_overlap_momentum_basis(overlap)
    print(calculate_max_overlap_transform(overlap_transform))  # noqa: T201


def load_overlap_nickel(
    i: int, j: int, offset: tuple[int, int] = (0, 0)
) -> Overlap[PositionBasisConfig[int, int, int]]:
    dx0, dx1 = offset
    i, j = (i, j) if i < j else (j, i)
    dx0, dx1 = (dx0 % 3, dx1 % 3) if i < j else ((-dx0) % 3, (-dx1) % 3)
    match (dx0, dx1):
        case (0, 2) | (1, 0) | (1, 2):
            dx0, dx1 = dx1, dx0

    path = get_data_path(f"overlap/overlap_{i}_{j}_{dx0}_{dx1}.npy")
    return load_overlap(path)


def print_max_and_min_overlap() -> None:
    for i in range(6):
        for j in range(i + 1, 6):
            overlap = load_overlap_nickel(i, j)
            overlap_transform = convert_overlap_momentum_basis(overlap)
            print(f"i={i}, j={j}")  # noqa: T201
            max_transform = calculate_max_overlap_transform(overlap_transform)
            print(max_transform)  # noqa: T201
            print(  # noqa: T201
                np.abs(max_transform[0]), np.linalg.norm(max_transform[1])
            )
            k0_transform = overlap_transform["vector"][0]
            print(k0_transform)  # noqa: T201
            print(np.abs(k0_transform))  # noqa: T201


def plot_all_abs_overlap_k() -> None:
    for i in range(6):
        for j in range(i + 1, 6):
            overlap = load_overlap_nickel(i, j)
            overlap_transform = convert_overlap_momentum_basis(overlap)

            fig, ax, _ = plot_overlap_k0k1(overlap_transform, 0, measure="abs")
            ax.set_title(f"Plot of the overlap transform for k2=0\nfor i={i} and j={j}")
            save_figure(fig, f"overlap/abs_overlap_k0k1_plot_{i}_{j}.png")
            fig.show()
    input()


def load_average_band_energies(
    n_bands: _L0Inv,
) -> np.ndarray[tuple[_L0Inv], np.dtype[np.float_]]:
    energies = np.zeros((n_bands,))
    for band in range(n_bands):
        wavepacket = load_nickel_wavepacket(band)
        energies[band] = wavepacket["energies"][0, 0]
    return energies  # type: ignore[no-any-return]


@npy_cached(lambda n_bands: get_data_path(f"incoherent_matrix_{n_bands}_bands.npy"))
def build_incoherent_matrix(
    n_bands: _L0Inv,
) -> np.ndarray[tuple[_L0Inv, _L0Inv, Literal[9]], np.dtype[np.float_]]:
    # The coefficients np.ndarray[tuple[_L0Inv, _L0Inv, Literal[9]], np.dtype[np.float_]]
    # represent the total rate R[i,j,dx] from i to j with an offset of dx at the location i.
    energies = load_average_band_energies(n_bands)
    out = np.zeros((n_bands, n_bands, 9))
    for i, j, dx0, dx1 in itertools.product(
        range(n_bands), range(n_bands), range(-1, 2), range(-1, 2)
    ):
        offset = (dx0, dx1)
        print(f"i={i}, j={j} offset={offset}")  # noqa: T201
        overlap = load_overlap_nickel(i, j, offset)
        overlap_transform = convert_overlap_momentum_basis(overlap)

        max_overlap, _ = calculate_max_overlap_transform(overlap_transform)
        exponential = np.exp(-(energies[j] - energies[i]) / (Boltzmann * 150))

        dx = np.ravel_multi_index(offset, (3, 3), mode="wrap")
        out[i, j, dx] = np.abs(max_overlap) ** 2 * exponential

    return out  # type: ignore [no-any-return]


def build_incoherent_matrix_fcc_hcp() -> (
    np.ndarray[tuple[Literal[2], Literal[2], Literal[9]], np.dtype[np.float_]]
):
    # The coefficients np.ndarray[tuple[_L0Inv, _L0Inv, Literal[9]], np.dtype[np.float_]]
    # represent the total rate R[i,j,dx] from i to j with an offset of dx at the location i.
    energies = load_average_band_energies(2)
    out = np.zeros((2, 2, 9))
    rate = 2.56410914e-05
    exponential = np.exp(-(energies[0] - energies[1]) / (Boltzmann * 150))
    print(exponential)  # noqa: T201
    out[0, 1, 0] = rate * exponential
    out[0, 1, np.ravel_multi_index((-1, 0), (3, 3), mode="wrap")] = rate * exponential
    out[0, 1, np.ravel_multi_index((0, -1), (3, 3), mode="wrap")] = rate * exponential
    out[1, 0, 0] = rate / exponential
    out[1, 0, np.ravel_multi_index((1, 0), (3, 3), mode="wrap")] = rate / exponential
    out[1, 0, np.ravel_multi_index((0, 1), (3, 3), mode="wrap")] = rate / exponential
    return out  # type: ignore [no-any-return]


def simulate_hydrogen_system() -> None:
    n_states = 6
    coefficients = build_incoherent_matrix(n_states)
    grid_shape = (10, 10)
    matrix = build_tunnelling_matrix(coefficients, grid_shape)

    times = np.array([0, 1, 2, 3, 4, 5]) * 2e4
    start = np.zeros(np.prod(grid_shape) * n_states)
    start[0] = 1
    out = simulate_tunnelling(matrix, start, times)

    print(np.sum(np.abs(out[:, -1])), np.sum(out[:, -1]))  # noqa: T201
    print(  # noqa: T201
        np.sum(np.abs(out[:, -1].reshape(*grid_shape, n_states)[:, :, 0])),
        np.sum(out[:, -1].reshape(*grid_shape, n_states)[:, :, 0]),
        np.sum(np.abs(out[:, -1].reshape(*grid_shape, n_states)[:, :, 1])),
        np.sum(out[:, -1].reshape(*grid_shape, n_states)[:, :, 1]),
    )

    fig, ax = plt.subplots()
    for i in range(start.size):
        ax.plot(times, out[i])
    ax.set_title("Plot of state occupation against time")
    fig.show()
    input()
