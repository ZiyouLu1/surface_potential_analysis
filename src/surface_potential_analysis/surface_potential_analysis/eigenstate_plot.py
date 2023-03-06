from typing import Literal

import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D

from surface_potential_analysis.brillouin_zone import grid_space

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
        mesh.set_norm(norm)  # type: ignore
        mesh.set_clim(0, max_clim)
    mesh0.set_norm(norm)  # type: ignore
    mesh0.set_clim(0, max_clim)

    ani = matplotlib.animation.ArtistAnimation(fig, frames)
    ax.set_xlabel("X direction")
    ax.set_ylabel("Y direction")
    fig.colorbar(mesh0, ax=ax, format="%4.1e")

    return fig, ax, ani


def plot_eigenstate_z(
    config: EigenstateConfig,
    eigenstate: Eigenstate,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    fig, a = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    util = EigenstateConfigUtil(config)

    z_points = np.linspace(-util.characteristic_z * 2, util.characteristic_z * 2, 1000)
    points = np.array(
        [(util.delta_x0[0] / 2, util.delta_x1[1] / 2, z) for z in z_points]
    )

    wfn = np.abs(util.calculate_wavefunction_fast(eigenstate, points))
    (line,) = a.plot(z_points, wfn)

    return fig, a, line


def plot_eigenstate_through_bridge(
    config: EigenstateConfig,
    eigenstate: Eigenstate,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs", "angle"] = "abs",
) -> tuple[Figure, Axes, Line2D]:
    fig, ax1 = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    util = EigenstateConfigUtil(config)

    x_points = np.linspace(0, util.delta_x0[0], 1000)
    points = np.array([(x, util.delta_x1[1] / 2, 0) for x in x_points])
    wfn = util.calculate_wavefunction_fast(eigenstate, points)

    if measure == "real":
        data = np.real(wfn)
    elif measure == "imag":
        data = np.imag(wfn)
    elif measure == "angle":
        data = np.unwrap(np.angle(wfn))
    else:
        data = np.abs(wfn)

    (line,) = ax1.plot(x_points - util.delta_x0[0] / 2, data)

    return fig, ax1, line


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


def plot_eigenstate_in_xz(
    config: EigenstateConfig,
    eigenstate: Eigenstate,
    ax: Axes | None = None,
    *,
    y_point=0.0,
    measure: Literal["real", "imag", "abs"] = "abs",
) -> tuple[Figure, Axes, AxesImage]:
    fig, ax1 = (ax.get_figure(), ax) if ax is not None else plt.subplots()
    util = EigenstateConfigUtil(config)

    x_points = np.linspace(0, util.delta_x0[0], 29, endpoint=False)
    z_lim = 2 * util.characteristic_z
    z_points = np.linspace(-z_lim, z_lim, 29, endpoint=True)

    xv, zv = np.meshgrid(x_points, z_points)
    points = np.array([xv.ravel(), y_point * np.ones_like(xv.ravel()), zv.ravel()]).T

    wfn = util.calculate_wavefunction_fast(eigenstate, points).reshape(xv.shape)
    match measure:
        case "real":
            data = np.real(wfn)
        case "imag":
            data = np.imag(wfn)
        case "abs":
            data = np.abs(wfn)
    im = ax1.imshow(data[:, ::-1])
    im.set_extent((x_points[0], x_points[-1], z_points[0], z_points[-1]))
    return (fig, ax1, im)


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


def plot_eigenstate(
    config: EigenstateConfig, eigenstate: Eigenstate, ax: Axes | None = None
) -> tuple[Figure, Axes]:

    fig, a = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    util = EigenstateConfigUtil(config)

    _, _, line = plot_eigenstate_z(config, eigenstate, ax)
    line.set_label("Z direction")

    _, _, line = plot_eigenstate_through_bridge(config, eigenstate, ax)
    line.set_label("X-Y through bridge")

    x_points = np.linspace(0, util.delta_x0[0], 1000)
    points = np.array([(x, x, 0) for x in x_points])
    a.plot(
        np.sqrt(2) * (x_points - util.delta_x0[0] / 2),
        np.abs(util.calculate_wavefunction_fast(eigenstate, points)),
        label="X-Y through Top",
    )
    a.legend()

    return (fig, a)
