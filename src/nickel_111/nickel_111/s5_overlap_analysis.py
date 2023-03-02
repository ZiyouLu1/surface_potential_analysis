from typing import Literal, Tuple

import numpy as np
import scipy.optimize
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from surface_potential_analysis.surface_config import (
    SurfaceConfig,
    get_surface_xy_points,
)
from surface_potential_analysis.surface_config_plot import (
    plot_points_on_surface_x0z,
    plot_points_on_surface_xy,
)

from .s5_overlap import OverlapTransform, load_overlap_transform
from .surface_data import get_data_path, save_figure


def plot_overlap_transform_xy(
    overlap: OverlapTransform,
    ikz=0,
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
    norm: Literal["symlog", "linear"] = "symlog",
) -> Tuple[Figure, Axes, QuadMesh]:
    reciprocal_surface: SurfaceConfig = {
        "delta_x0": overlap["dkx0"] * overlap["points"].shape[0],
        "delta_x1": overlap["dkx1"] * overlap["points"].shape[1],
    }

    fig, ax, mesh = plot_points_on_surface_xy(
        reciprocal_surface,
        np.fft.fftshift(overlap["points"], axes=(0, 1)),
        z_ind=ikz,
        ax=ax,
        measure=measure,
    )
    ax.set_xlabel("kx direction")
    ax.set_ylabel("ky direction")
    mesh.set_norm(norm)  # type: ignore
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(mesh, ax=ax, format="%4.1e")

    return fig, ax, mesh


def plot_overlap_transform_x0z(
    overlap: OverlapTransform,
    x1_ind: int = 0,
    *,
    measure: Literal["real", "imag", "abs"] = "abs",
    ax: Axes | None = None,
    norm: Literal["symlog", "linear"] = "symlog",
) -> Tuple[Figure, Axes, QuadMesh]:
    reciprocal_surface: SurfaceConfig = {
        "delta_x0": overlap["dkx0"] * overlap["points"].shape[0],
        "delta_x1": overlap["dkx1"] * overlap["points"].shape[1],
    }
    z_points = overlap["dkz"] * np.arange(overlap["points"].shape[2])

    fig, ax, mesh = plot_points_on_surface_x0z(
        reciprocal_surface,
        np.fft.fftshift(overlap["points"], axes=(0,)),
        z_points.tolist(),
        x1_ind=x1_ind,
        ax=ax,
        measure=measure,
    )

    ax.set_xlabel("kx0 direction")
    ax.set_ylabel("kz direction")
    mesh.set_norm(norm)  # type: ignore
    # ax.set_aspect("equal", adjustable="box")
    fig.colorbar(mesh, ax=ax, format="%4.1e")

    return fig, ax, mesh


def plot_overlap_transform_along_diagonal(
    overlap: OverlapTransform,
    kz_ind: int = 0,
    *,
    measure: Literal["real", "imag", "abs", "angle"] = "abs",
    ax: Axes | None = None,
) -> Tuple[Figure, Axes, Line2D]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    points = np.diag(overlap["points"][:, :, kz_ind])
    if measure == "real":
        data = np.real(points)
    elif measure == "imag":
        data = np.imag(points)
    elif measure == "abs":
        data = np.abs(points)
    else:
        data = np.unwrap(np.angle(points))

    kx1_points = np.fft.fftfreq(data.shape[0], np.linalg.norm(overlap["dkx1"]))
    (line,) = ax.plot(np.fft.fftshift(kx1_points), np.fft.fftshift(data))
    ax.set_xlabel("kx1 direction")

    return fig, ax, line


def plot_overlap():
    path = get_data_path("overlap_transform_hcp_fcc.npz")
    overlap = load_overlap_transform(path)

    fig, ax, _ = plot_overlap_transform_xy(overlap)
    ax.set_title(
        "Plot of the overlap transform for ikz=0\n"
        "showing oscillation in the direction corresponding to\n"
        "a vector spanning the fcc and hcp sites"
    )
    save_figure(fig, "2d_overlap_fraction_kx_ky.png")
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

    print(points.shape)
    print((points[177, 177, 0]))
    print((points[177, 178, 0]))
    print((points[178, 177, 0]))
    print((points[178, 178, 0]))

    print(np.unravel_index(np.argmax(points), shape=points.shape))
    (ikx0, ikx1, _) = np.unravel_index(np.argmax(points), shape=points.shape)
    ikx0 = 184 - ikx0
    ikx1 = 184 - ikx1
    kmax = (
        (overlap["dkx0"][0] * ikx0) + (overlap["dkx1"][0] * ikx1),
        (overlap["dkx0"][1] * ikx0) + (overlap["dkx1"][1] * ikx1),
    )
    print(kmax, np.linalg.norm(kmax))

    def fit_curve(
        kxy: Tuple[NDArray, NDArray], dkx: float, dky: float, A0: float, alpha: float
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
