import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .energy_data import EnergyInterpolation
from .energy_eigenstate import EigenstateConfig


def plot_interpolation_with_sho(
    interpolation: EnergyInterpolation,
    eigenstate_config: EigenstateConfig,
    z_offset: float,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    fig, a = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    points = np.array(interpolation["points"])
    middle_x_index = math.floor(points.shape[0] / 2)
    middle_y_index = math.floor(points.shape[1] / 2)
    start_z = z_offset
    end_z = interpolation["dz"] * (points.shape[2] - 1) + z_offset
    z_points = np.linspace(start_z, end_z, points.shape[2])

    a.plot(z_points, points[middle_x_index, middle_y_index])
    sho_pot = (
        0.5
        * eigenstate_config["mass"]
        * (eigenstate_config["sho_omega"] * z_points) ** 2
    )
    a.plot(z_points, sho_pot)

    max_potential = 1e-18
    a.set_ylim(0, max_potential)

    return fig, a
