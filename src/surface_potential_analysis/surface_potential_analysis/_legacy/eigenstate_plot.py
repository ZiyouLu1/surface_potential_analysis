from typing import Literal

import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from surface_potential_analysis._legacy.brillouin_zone import grid_space
from surface_potential_analysis._legacy.energy_data_plot import (
    calculate_cumulative_distances_along_path,
)
from surface_potential_analysis._legacy.wavepacket_grid import WavepacketGrid
from surface_potential_analysis._legacy.wavepacket_grid_plot import (
    plot_wavepacket_grid_x0z,
)
from surface_potential_analysis.interpolation import interpolate_points_fftn

from .eigenstate import Eigenstate, EigenstateConfig, EigenstateConfigUtil


def plot_eigenstate_in_xy(
    config: EigenstateConfig,
    eigenstate: Eigenstate,
    z_point=0.0,
    *,
    shape: tuple[int, int] = (29, 29),
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
) -> tuple[Figure, Axes, QuadMesh]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()
    util = EigenstateConfigUtil(config)

    xy_coordinates = grid_space(config["delta_x0"], config["delta_x1"], shape)

    points = np.array(
        [
            xy_coordinates[:, 0],
            xy_coordinates[:, 1],
            z_point * np.ones_like(xy_coordinates[:, 0]),
        ]
    ).T

    wfn = util.calculate_wavefunction_fast(eigenstate, points).reshape(shape)
    match measure:
        case "real":
            data = np.real(wfn)
        case "imag":
            data = np.imag(wfn)
        case "abs":
            data = np.abs(wfn)

    mesh = ax.pcolormesh(
        xy_coordinates[:, 0].reshape(shape),
        xy_coordinates[:, 1].reshape(shape),
        data,
        shading="nearest",
    )
    return (fig, ax, mesh)


def animate_eigenstate_3D_in_xy(
    config: EigenstateConfig,
    eigenstate: Eigenstate,
    *,
    shape: tuple[int, int, int] = (29, 29, 20),
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
    norm: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, matplotlib.animation.ArtistAnimation]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    util = EigenstateConfigUtil(config)
    z_points = np.linspace(
        -util.characteristic_z * 2, util.characteristic_z * 3, shape[2]
    )

    _, _, mesh0 = plot_eigenstate_in_xy(
        config,
        eigenstate,
        z_points[0],
        shape=(shape[0], shape[1]),
        ax=ax,
        measure=measure,
    )

    frames: list[list[QuadMesh]] = []
    for z_point in z_points:
        _, _, mesh = plot_eigenstate_in_xy(
            config,
            eigenstate,
            z_point,
            shape=(shape[0], shape[1]),
            ax=ax,
            measure=measure,
        )

        frames.append([mesh])

    max_clim = np.max([i[0].get_clim()[1] for i in frames])
    for (mesh,) in frames:
        mesh.set_norm(norm)
        mesh.set_clim(0, max_clim)
    mesh0.set_norm(norm)
    mesh0.set_clim(0, max_clim)

    ani = matplotlib.animation.ArtistAnimation(fig, frames)
    ax.set_xlabel("X direction")
    ax.set_ylabel("Y direction")
    fig.colorbar(mesh0, ax=ax, format="%4.1e")

    return fig, ax, ani


def plot_eigenstate_along_path(
    config: EigenstateConfig,
    eigenstate: Eigenstate,
    points: NDArray,
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
) -> tuple[Figure, Axes, Line2D]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    util = EigenstateConfigUtil(config)
    wfn = np.abs(util.calculate_wavefunction_fast(eigenstate, points))

    match measure:
        case "real":
            data = np.real(wfn)
        case "imag":
            data = np.imag(wfn)
        case "abs":
            data = np.abs(wfn)
    distances = calculate_cumulative_distances_along_path(
        np.arange(points.shape[0]).reshape(1, -1), points
    )
    (line,) = ax.plot(distances, data)
    return fig, ax, line


def plot_wavefunction_difference_in_xy(
    config: EigenstateConfig,
    eigenstate1: Eigenstate,
    eigenstate2: Eigenstate,
    ax: Axes | None = None,
    y_point=0.0,
) -> tuple[Figure, Axes, AxesImage]:
    fig, ax1 = (ax.get_figure(), ax) if ax is not None else plt.subplots()
    util = EigenstateConfigUtil(config)

    x_points = np.linspace(0, util.delta_x0[0], 30)
    y_points = np.linspace(0, util.delta_x1[1], 30)

    xv, yv = np.meshgrid(x_points, y_points)
    points = np.array([xv.ravel(), yv.ravel(), y_point * np.ones_like(xv.ravel())]).T

    wfn1 = util.calculate_wavefunction_fast(eigenstate1, points).reshape(xv.shape)
    wfn2 = util.calculate_wavefunction_fast(eigenstate2, points).reshape(xv.shape)
    X = np.abs(wfn1) - np.abs(wfn2)

    im = ax1.imshow(np.abs(X))
    im.set_extent((x_points[0], x_points[-1], y_points[0], y_points[-1]))
    return (fig, ax1, im)


def plot_bloch_wavefunction_difference_in_x0z(
    config0: EigenstateConfig,
    eigenstate0: Eigenstate,
    config1: EigenstateConfig,
    eigenstate1: Eigenstate,
    x1_ind: int = 0,
    z_points: list[float] | None = None,
    *,
    measure: Literal["real", "imag", "abs"] = "abs",
    norm: Literal["symlog", "linear"] = "linear",
    ax: Axes | None = None,
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot the difference between the bloch wavefunction for the two eigenstates
    at x1=x1_ind, where the

    Parameters
    ----------
    config0 : EigenstateConfig
    eigenstate0 : Eigenstate
    config1: EigenstateConfig
    eigenstate1 : Eigenstate
    x1_ind : int, optional
        Index of the x1 plane to plot, by default 0.
        Index is taken on the interpolated grid of size NX1_0 * NX1_1
    z_points : list[float] | None, optional
        z_points, by default np.linspace(-1 * util.characteristic_z, 4 * util.characteristic_z, 50)
    measure : Literal[&quot;real&quot;, &quot;imag&quot;, &quot;abs&quot;], optional
        The measure to display on the quad mesh, by default "abs"
    ax : Axes | None, optional
        axis to plot on, by default None

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
        The figure, axis and QuadMesh for the final image
    """
    util0 = EigenstateConfigUtil(config0)

    z_points_actual: list[float] = (
        np.linspace(
            -1 * util0.characteristic_z, 4 * util0.characteristic_z, 50
        ).tolist()
        if z_points is None
        else z_points
    )

    wfn0 = util0.calculate_bloch_wavefunction_fourier(
        eigenstate0["eigenvector"], z_points_actual
    )
    util1 = EigenstateConfigUtil(config1)
    wfn1 = util1.calculate_bloch_wavefunction_fourier(
        eigenstate1["eigenvector"], z_points_actual
    )

    wfn0_interpolated = interpolate_points_fftn(
        wfn0, s=np.multiply(wfn1.shape[0:2], wfn0.shape[0:2]).tolist(), axes_arr=(0, 1)
    )
    wfn1_interpolated = interpolate_points_fftn(
        wfn1, s=np.multiply(wfn1.shape[0:2], wfn0.shape[0:2]).tolist(), axes_arr=(0, 1)
    )

    grid: WavepacketGrid = {
        "delta_x0": util0.delta_x0,
        "delta_x1": util0.delta_x1,
        "points": wfn0_interpolated - wfn1_interpolated,
        "z_points": z_points_actual,
    }
    return plot_wavepacket_grid_x0z(grid, x1_ind, measure=measure, norm=norm, ax=ax)


def plot_eigenstate_x0z(
    config: EigenstateConfig,
    eigenstate: Eigenstate,
    x1_ind: int = 0,
    z_points: list[float] | None = None,
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
    norm: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, AxesImage]:
    util = EigenstateConfigUtil(config)

    z_points_actual: list[float] = (
        np.linspace(-1 * util.characteristic_z, 4 * util.characteristic_z, 50).tolist()
        if z_points is None
        else z_points
    )

    wfn = util.calculate_bloch_wavefunction_fourier(
        eigenstate["eigenvector"], z_points_actual
    )

    grid: WavepacketGrid = {
        "delta_x0": util.delta_x0,
        "delta_x1": util.delta_x1,
        "points": wfn.tolist(),
        "z_points": z_points_actual,
    }
    return plot_wavepacket_grid_x0z(grid, x1_ind, measure=measure, norm=norm, ax=ax)


def plot_eigenstate_in_yz(
    config: EigenstateConfig,
    eigenstate: Eigenstate,
    ax: Axes | None = None,
    *,
    x_point=0.0,
    measure: Literal["real", "imag", "abs"] = "abs",
) -> tuple[Figure, Axes, AxesImage]:
    fig, ax1 = (ax.get_figure(), ax) if ax is not None else plt.subplots()
    util = EigenstateConfigUtil(config)

    y_points = np.linspace(0, util.delta_x1[1], 29, endpoint=False)
    z_lim = 2 * util.characteristic_z
    z_points = np.linspace(-z_lim, z_lim, 29, endpoint=True)

    yv, zv = np.meshgrid(y_points, z_points)
    points = np.array([x_point * np.ones_like(yv.ravel()), yv.ravel(), zv.ravel()]).T

    wfn = util.calculate_wavefunction_fast(eigenstate, points).reshape(yv.shape)
    match measure:
        case "real":
            data = np.real(wfn)
        case "imag":
            data = np.imag(wfn)
        case "abs":
            data = np.abs(wfn)
    im = ax1.imshow(data[:, ::-1])
    im.set_extent((y_points[0], y_points[-1], z_points[0], z_points[-1]))
    return (fig, ax1, im)
