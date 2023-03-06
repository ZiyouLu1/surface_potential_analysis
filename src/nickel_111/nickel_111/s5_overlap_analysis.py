import numpy as np
import scipy.optimize
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from surface_potential_analysis.energy_data_plot import (
    calculate_cumulative_distances_along_path,
)
from surface_potential_analysis.surface_config import (
    SurfaceConfig,
    get_surface_xy_points,
)
from surface_potential_analysis.surface_config_plot import (
    plot_ft_points_on_surface_xy,
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
) -> tuple[Figure, Axes, QuadMesh]:
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


def plot_overlap_xy(
    overlap: OverlapTransform,
    ikz=0,
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
    norm: Literal["symlog", "linear"] = "symlog",
) -> tuple[Figure, Axes, QuadMesh]:
    reciprocal_surface: SurfaceConfig = {
        "delta_x0": overlap["dkx0"] * overlap["points"].shape[0],
        "delta_x1": overlap["dkx1"] * overlap["points"].shape[1],
    }

    fig, ax, mesh = plot_ft_points_on_surface_xy(
        reciprocal_surface,
        overlap["points"],
        z_ind=ikz,
        ax=ax,
        measure=measure,
    )
    ax.set_xlabel("x direction")
    ax.set_ylabel("y direction")
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
) -> tuple[Figure, Axes, QuadMesh]:
    reciprocal_surface: SurfaceConfig = {
        "delta_x0": overlap["dkx0"] * overlap["points"].shape[0],
        "delta_x1": overlap["dkx1"] * overlap["points"].shape[1],
    }
    z_points = overlap["dkz"] * np.arange(overlap["points"].shape[2])

    fig, ax, mesh = plot_points_on_surface_x0z(
        reciprocal_surface,
        np.fft.fftshift(overlap["points"], axes=(0,)).tolist(),
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


def plot_overlap_transform_along_path(
    overlap: OverlapTransform,
    path: NDArray,
    kz_ind: int = 0,
    *,
    measure: Literal["real", "imag", "abs", "angle"] = "abs",
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    overlap_shape = np.shape(overlap["points"])
    points = np.fft.fftshift(overlap["points"], axes=(0, 1))[
        path[:, 0], path[:, 1], kz_ind
    ]
    if measure == "real":
        data = np.real(points)
    elif measure == "imag":
        data = np.imag(points)
    elif measure == "abs":
        data = np.abs(points)
    else:
        data = np.unwrap(np.angle(points))

    kxy_points = get_surface_xy_points(
        {
            "delta_x0": (
                overlap["dkx0"][0] * overlap_shape[0],
                overlap["dkx0"][1] * overlap_shape[0],
            ),
            "delta_x1": (
                overlap["dkx1"][0] * overlap_shape[1],
                overlap["dkx1"][1] * overlap_shape[1],
            ),
        },
        shape=(overlap_shape[0], overlap_shape[1]),
    )

    distances = calculate_cumulative_distances_along_path(path, kxy_points)
    (line,) = ax.plot(distances, data)
    return fig, ax, line


def plot_overlap_transform_along_diagonal(
    overlap: OverlapTransform,
    kz_ind: int = 0,
    *,
    measure: Literal["real", "imag", "abs", "angle"] = "abs",
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:

    path = np.array([[i, i] for i in range(overlap["points"].shape[0])])
    return plot_overlap_transform_along_path(
        overlap, path, kz_ind, measure=measure, ax=ax
    )


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


def calculate_max_overlap_transform(overlap: OverlapTransform):
    points = overlap["points"]
    xy_points = get_surface_xy_points(
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
    (ikx0, ikx1, ikz) = np.unravel_index(np.argmax(np.abs(points)), shape=points.shape)
    xy_points -= xy_points[points.shape[0] // 2, points.shape[1] // 2]

    xy_point = xy_points[ikx0, ikx1]
    return points[ikx0, ikx1, ikz], xy_point


def print_max_overlaps():
    path = get_data_path("overlap_transform_hcp_fcc.npz")
    overlap = load_overlap_transform(path)
    print(calculate_max_overlap_transform(overlap))

    path = get_data_path("overlap_transform_fcc_fcc.npz")
    overlap = load_overlap_transform(path)
    print(calculate_max_overlap_transform(overlap))

    path = get_data_path("overlap_transform_hcp_hcp.npz")
    overlap = load_overlap_transform(path)
    print(calculate_max_overlap_transform(overlap))
