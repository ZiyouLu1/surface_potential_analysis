from typing import List, Literal

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from .surface_config import SurfaceConfig, get_reciprocal_surface, get_surface_xy_points


def plot_points_on_surface_xy(
    surface: SurfaceConfig,
    points: List[List[List[complex]]],
    z_ind=0,
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
):
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    if measure == "real":
        data = np.real(points)
    elif measure == "imag":
        data = np.imag(points)
    else:
        data = np.abs(points)

    coordinates = get_surface_xy_points(
        surface, (data.shape[0], data.shape[1])
    ).reshape(data.shape[0], data.shape[1], 2)
    mesh = ax.pcolormesh(
        coordinates[:, :, 0], coordinates[:, :, 1], data[:, :, z_ind], shading="nearest"
    )

    return fig, ax, mesh


def plot_ft_points_on_surface_xy(
    surface: SurfaceConfig,
    points: List[List[List[complex]]],
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
