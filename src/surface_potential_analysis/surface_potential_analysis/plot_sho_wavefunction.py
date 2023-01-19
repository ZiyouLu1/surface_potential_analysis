import math
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .energy_data import EnergyInterpolation
from .energy_eigenstate import EigenstateConfig


def plot_energy_with_sho_potential(
    interpolation: EnergyInterpolation,
    eigenstate_config: EigenstateConfig,
    z_offset: float,
    xy_ind: Tuple[int, int],
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    fig, a = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    points = np.array(interpolation["points"])
    start_z = z_offset
    end_z = interpolation["dz"] * (points.shape[2] - 1) + z_offset
    z_points = np.linspace(start_z, end_z, points.shape[2])

    (line1,) = a.plot(z_points, points[xy_ind[0], xy_ind[1]])
    line1.set_label("Potential at center point")

    sho_pot = (
        0.5
        * eigenstate_config["mass"]
        * (eigenstate_config["sho_omega"] * z_points) ** 2
    )
    (line2,) = a.plot(z_points, sho_pot)
    line2.set_label("SHO Config")

    return fig, a


def plot_energy_with_sho_potential_at_hollow(
    interpolation: EnergyInterpolation,
    eigenstate_config: EigenstateConfig,
    z_offset: float,
    ax: Axes | None = None,
) -> Tuple[Figure, Axes]:
    points = np.array(interpolation["points"])
    middle_x_index = math.floor(points.shape[0] / 2)
    middle_y_index = math.floor(points.shape[1] / 2)
    xy_ind = (middle_x_index, middle_y_index)

    fig, ax = plot_energy_with_sho_potential(
        interpolation, eigenstate_config, z_offset, xy_ind, ax
    )

    max_potential = 1e-18
    ax.set_ylim(0, max_potential)
    return fig, ax


def plot_energy_with_sho_potential_at_minimum(
    interpolation: EnergyInterpolation,
    eigenstate_config: EigenstateConfig,
    ax: Axes | None = None,
) -> Tuple[Figure, Axes]:
    points = np.array(interpolation["points"], dtype=float)
    arg_min = np.unravel_index(np.argmin(points), points.shape)
    xy_ind = (arg_min[0], arg_min[1])
    print(arg_min)
    z_offset = -interpolation["dz"] * arg_min[2]

    fig, ax = plot_energy_with_sho_potential(
        interpolation, eigenstate_config, z_offset, xy_ind, ax
    )

    max_potential = 1e-18
    ax.set_ylim(0, max_potential)
    return fig, ax
