from typing import List, Literal, Sequence, Tuple

import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from .energy_eigenstate import Eigenstate, EigenstateConfig, EigenstateConfigUtil


def get_eigenstate_frame(
    data: NDArray, ax: Axes, clim: Tuple[float, float], extent: Sequence[float]
) -> AxesImage:
    img = ax.imshow(data)
    img.set_extent(extent)
    img.set_clim(*clim)
    img.set_norm("symlog")  # type: ignore
    return img


def plot_eigenstate_3D(
    config: EigenstateConfig, eigenstate: Eigenstate, ax: Axes | None = None
) -> tuple[Figure, Axes, matplotlib.animation.ArtistAnimation]:
    fig, ax1 = (ax.get_figure(), ax) if ax is not None else plt.subplots()
    util = EigenstateConfigUtil(config)

    x_points = np.linspace(0, util.delta_x, 50)
    y_points = np.linspace(0, util.delta_y, 50)
    z_points = np.linspace(-util.delta_x / 2, util.delta_x / 2, 20)

    xv, yv, zv = np.meshgrid(x_points, y_points, z_points)

    points = np.array([xv.ravel(), yv.ravel(), zv.ravel()]).T
    wfn = util.calculate_wavefunction_fast(eigenstate, points).reshape(xv.shape)
    data = np.abs(wfn)

    extent = [x_points[0], x_points[-1], y_points[0], y_points[-1]]
    clim = (np.min(data), np.max(data))

    get_eigenstate_frame(data[:, :, 0], ax1, clim, extent)

    ims: List[List[AxesImage]] = []
    for z_ind in range(z_points.shape[0]):

        img = get_eigenstate_frame(data[:, :, z_ind], ax1, clim, extent)
        ims.append([img])

    ani = matplotlib.animation.ArtistAnimation(fig, ims)

    return fig, ax1, ani


def plot_eigenstate_z(
    config: EigenstateConfig,
    eigenstate: Eigenstate,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    fig, a = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    util = EigenstateConfigUtil(config)

    z_points = np.linspace(-util.delta_x / 2, util.delta_x / 2, 1000)
    points = np.array([(util.delta_x / 2, util.delta_y / 2, z) for z in z_points])

    wfn = np.abs(util.calculate_wavefunction_fast(eigenstate, points))
    (line,) = a.plot(z_points, wfn)

    return fig, a, line


def plot_eigenstate_through_bridge(
    config: EigenstateConfig,
    eigenstate: Eigenstate,
    ax: Axes | None = None,
    view: Literal["abs"] | Literal["angle"] = "abs",
) -> tuple[Figure, Axes, Line2D]:
    fig, ax1 = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    util = EigenstateConfigUtil(config)

    x_points = np.linspace(0, util.delta_x, 1000)
    points = np.array([(x, util.delta_y / 2, 0) for x in x_points])
    wfn = util.calculate_wavefunction_fast(eigenstate, points)
    (line,) = ax1.plot(
        x_points - util.delta_x / 2,
        np.abs(wfn) if view == "abs" else np.angle(wfn),
    )

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

    x_points = np.linspace(0, util.delta_x, 30)
    y_points = np.linspace(0, util.delta_y, 30)

    xv, yv = np.meshgrid(x_points, y_points)
    points = np.array([xv.ravel(), yv.ravel(), y_point * np.ones_like(xv.ravel())]).T

    wfn1 = util.calculate_wavefunction_fast(eigenstate1, points).reshape(xv.shape)
    wfn2 = util.calculate_wavefunction_fast(eigenstate2, points).reshape(xv.shape)
    X = np.abs(wfn1) - np.abs(wfn2)

    im = ax1.imshow(np.abs(X))
    im.set_extent((x_points[0], x_points[-1], y_points[0], y_points[-1]))
    return (fig, ax1, im)


def plot_eigenstate_in_xy(
    config: EigenstateConfig,
    eigenstate: Eigenstate,
    ax: Axes | None = None,
    y_point=0.0,
) -> tuple[Figure, Axes, AxesImage]:
    fig, ax1 = (ax.get_figure(), ax) if ax is not None else plt.subplots()
    util = EigenstateConfigUtil(config)

    x_points = np.linspace(0, util.delta_x, 30)
    y_points = np.linspace(0, util.delta_y, 30)

    xv, yv = np.meshgrid(x_points, y_points)
    points = np.array([xv.ravel(), yv.ravel(), y_point * np.ones_like(xv.ravel())]).T

    X = util.calculate_wavefunction_fast(eigenstate, points).reshape(xv.shape)
    im = ax1.imshow(np.abs(X))
    im.set_extent((x_points[0], x_points[-1], y_points[0], y_points[-1]))
    return (fig, ax1, im)
