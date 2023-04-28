from __future__ import annotations

from typing import Literal, TypeVar

import numpy as np
import scipy.optimize
from matplotlib import pyplot as plt
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
    plot_overlap_2d,
    plot_overlap_transform_2d,
    plot_overlap_transform_along_diagonal,
)

from .surface_data import get_data_path, save_figure

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
    points = np.asarray(overlap["vector"])
    (ik0, ik1, inz) = np.unravel_index(np.argmax(np.abs(points)), shape=points.shape)
    return (int(ik0), int(ik1), int(inz))


def make_transform_real_at(
    overlap: OverlapTransform[_L0Inv, _L1Inv, _L2Inv],
    point: tuple[int, int, int] | int | None = None,
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
    util = BasisConfigUtil(overlap["basis"]).shape
    point = np.argmax(np.abs(overlap["vector"])) if point is None else point
    point = point if isinstance(point, int) else np.unravel_index(point, util.shape)

    new_points = overlap["vector"] * np.exp(-1j * np.angle(overlap["vector"][point]))
    return {
        "basis": overlap["basis"],
        "vector": new_points,
    }


def plot_fcc_hcp_overlap_transform() -> None:
    overlap = load_overlap_fcc_hcp()
    overlap_transform = convert_overlap_momentum_basis(overlap)

    fig, ax, _ = plot_overlap_transform_2d(overlap_transform, 0, 2, measure="abs")
    ax.set_title(
        "Plot of the overlap transform for ikz=0\n"
        "showing oscillation in the direction corresponding to\n"
        "a vector spanning the fcc and hcp sites"
    )
    save_figure(fig, "2d_overlap_transform_kx_ky.png")
    fig.show()

    fig, ax, _ = plot_overlap_transform_2d(overlap_transform, 0, 2, measure="real")
    ax.set_title(
        "Plot of the overlap transform for ikz=0\n"
        "showing oscillation in the direction corresponding to\n"
        "a vector spanning the fcc and hcp sites"
    )
    save_figure(fig, "2d_overlap_transform_real_kx_ky.png")
    fig.show()

    fig, ax, _ = plot_overlap_transform_2d(overlap_transform, 0, 2, measure="imag")
    ax.set_title(
        "Plot of the overlap transform for ikz=0\n"
        "showing oscillation in the direction corresponding to\n"
        "a vector spanning the fcc and hcp sites"
    )
    save_figure(fig, "2d_overlap_transform_imag_kx_ky.png")
    fig.show()

    fig, ax, _ = plot_overlap_transform_2d(overlap_transform, 0, 0)
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
    _, _, ln = plot_overlap_transform_along_diagonal(
        overlap_transform, 2, measure="abs", ax=ax
    )
    ln.set_label("abs")
    _, _, ln = plot_overlap_transform_along_diagonal(
        overlap_transform, 2, measure="real", ax=ax
    )
    ln.set_label("real")
    _, _, ln = plot_overlap_transform_along_diagonal(
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
    fig, ax, _ = plot_overlap_2d(overlap, 177, 2, measure="abs")
    ax.set_title(
        "Plot of the overlap summed over z\n"
        "showing the FCC and HCP asymmetry\n"
        "in a small region in the center of the figure"
    )
    save_figure(fig, "2d_overlap_kx_ky.png")
    fig.show()

    fig, ax, _ = plot_overlap_2d(overlap, 177, 2, measure="real")
    ax.set_title(
        "Plot of the overlap summed over z\n"
        "showing the FCC and HCP asymmetry\n"
        "in a small region in the center of the figure"
    )
    save_figure(fig, "2d_overlap_real_kx_ky.png")
    fig.show()
    input()


def _() -> None:
    delta_k = np.linalg.norm(
        np.add(
            np.multiply(overlap["dkx0"], overlap["points"].shape[0]),
            np.multiply(overlap["dkx1"], overlap["points"].shape[1]),
        )
    )
    k_points = np.linspace(0, delta_k, 5000)
    #     0.75
    #     1.1

    #     0.4

    #     -2

    ax2 = ax.twinx()
    _, _, ln = plot_overlap_transform_along_diagonal(overlap, measure="angle", ax=ax2)
    ln.set_label("angle")
    fit_points = (
        -2
        * ((k_points - (delta_k / 2)) / (delta_k / 8))
        * (0.72 - 0.35 * np.cos(4 * np.pi * (k_points - (delta_k / 2)) / (delta_k / 8)))
        * np.min(np.real(overlap["points"]))
        * np.exp(-((k_points - (delta_k / 2)) ** 2) / (delta_k / 7) ** 2)
    )

    def fit_curve(
        kxy: tuple[NDArray, NDArray], dkx: float, dky: float, A0: float, alpha: float
    ):
        (kx, ky) = kxy
        return (
            (np.sin(kx / dkx + ky / dky) ** 2)
            * A0
            * np.exp(-(kx**2 + ky**2) / alpha**2)
        )

    ax2.plot(k_points, fit_points)

    ax.legend()
    ax.set_title(
        "Plot of the wavefunction along the diagonal,\nshowing an oscillation in the overlap"
    )
    save_figure(fig, "diagonal_1d_overlap_fraction.png")
    fig.show()
    input()


def fit_overlap_transform():
    overlap = load_overlap_fcc_hcp()
    overlap_transform = convert_overlap_momentum_basis(overlap)
    points = overlap_transform["vector"]

    print(points[0, 0, 0])
    print(np.max(np.abs(points[:, :, 0])))
    print(np.max(np.abs(points[:, :])))

    print(points.shape)
    print(points[177, 177, 0])
    print(points[177, 178, 0])
    print(points[178, 177, 0])
    print(points[178, 178, 0])

    print(get_max_point(overlap))
    (ikx0, ikx1, _) = get_max_point(overlap)
    ikx0 = 184 - ikx0
    ikx1 = 184 - ikx1
    kmax = (
        (overlap["dkx0"][0] * ikx0) + (overlap["dkx1"][0] * ikx1),
        (overlap["dkx0"][1] * ikx0) + (overlap["dkx1"][1] * ikx1),
    )
    print(kmax, np.linalg.norm(kmax))

    def fit_curve(
        kxy: tuple[NDArray, NDArray], dkx: float, dky: float, A0: float, alpha: float
    ):
        (kx, ky) = kxy
        return (
            (np.sin(kx / dkx + ky / dky) ** 2)
            * A0
            * np.exp(-(kx**2 + ky**2) / alpha**2)
        )

    offset = (
        -overlap["dkx0"][0] * (points.shape[0] // 2)
        - overlap["dkx1"][0] * (points.shape[1] // 2),
        -overlap["dkx0"][1] * (points.shape[0] // 2)
        - overlap["dkx1"][1] * (points.shape[1] // 2),
    )

    print(offset)

    coordinates = get_surface_xy_points(
        {
            "delta_x0": (
                overlap["dkx0"][0] * points.shape[0],
                overlap["dkx0"][1] * points.shape[0],
            ),
            "delta_x1": (
                overlap["dkx1"][0] * points.shape[1],
                overlap["dkx1"][1] * points.shape[1],
            ),
        },
        shape=(points.shape[0], points.shape[1]),
    )
    # TODO: maybe ifftshift
    amplitudes = np.fft.fftshift(np.abs(points[:, :, 0]))

    (ikx0, ikx1) = np.unravel_index(np.argmax(amplitudes), shape=amplitudes.shape)
    print(ikx0, ikx1)
    print(coordinates[92, 92])
    print(coordinates[ikx0, ikx1])
    coordinates -= coordinates[92, 92]
    print(coordinates[92, 92])
    print(coordinates[ikx0, ikx1])

    kx_coord = coordinates[:, :, 0].ravel()
    ky_coord = coordinates[:, :, 1].ravel()
    popt, pcov = scipy.optimize.curve_fit(
        fit_curve,
        (kx_coord, ky_coord),
        amplitudes.ravel(),
        p0=[
            coordinates[ikx0, ikx1][0],
            coordinates[ikx0, ikx1][1],
            np.max(amplitudes),
            np.linalg.norm(overlap["dkx1"]),
        ],
    )
    print("popt", popt)
    print("pcov", pcov)

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
