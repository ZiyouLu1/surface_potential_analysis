from typing import Literal

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from .surface_config import SurfaceConfig, get_reciprocal_surface, get_surface_xy_points


def plot_points_on_surface_xy(
    surface: SurfaceConfig,
    points: list[list[list[complex]]],
    z_ind=0,
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs", "angle"] = "abs",
):
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    match measure:
        case "real":
            data = np.real(points)
        case "imag":
            data = np.imag(points)
        case "abs":
            data = np.abs(points)
        case "angle":
            data = np.unwrap(np.angle(points))

    coordinates = get_surface_xy_points(surface, (data.shape[0], data.shape[1]))
    mesh = ax.pcolormesh(
        coordinates[:, :, 0], coordinates[:, :, 1], data[:, :, z_ind], shading="nearest"
    )

    return fig, ax, mesh


def plot_ft_points_on_surface_xy(
    surface: SurfaceConfig,
    points: list[list[list[complex]]],
    z_ind=0,
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
):
    ft_points = np.fft.ifft2(points, axes=(0, 1)).tolist()
    ft_surface = get_reciprocal_surface(surface)
    return plot_points_on_surface_xy(
        ft_surface, ft_points, z_ind=z_ind, ax=ax, measure=measure
    )


def plot_points_on_surface_x0z(
    surface: SurfaceConfig,
    points: list[list[list[complex]]],
    z_points: list[float],
    x1_ind: int,
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs", "angle"] = "abs",
):
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    match measure:
        case "real":
            data = np.real(points)
        case "imag":
            data = np.imag(points)
        case "abs":
            data = np.abs(points)
        case "angle":
            data = np.unwrap(np.angle(points))

    x0_points = np.linspace(0, np.linalg.norm(surface["delta_x0"]), data.shape[0])
    x0v, zv = np.meshgrid(x0_points, z_points, indexing="ij")
    mesh = ax.pcolormesh(x0v, zv, data[:, x1_ind, :], shading="nearest")

    return fig, ax, mesh
