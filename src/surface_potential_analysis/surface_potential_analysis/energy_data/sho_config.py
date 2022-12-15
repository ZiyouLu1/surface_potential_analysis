import math
from typing import Tuple, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from .energy_data import EnergyInterpolation


class EigenstateConfig(TypedDict):
    sho_omega: float
    """Angular frequency (in rad s-1) of the sho we will fit using"""
    mass: float
    """Mass in Kg"""
    delta_x: float
    """maximum extent in the x direction"""
    delta_y: float
    """maximum extent in the x direction"""


def get_minimum_coordinate(arr: ArrayLike) -> Tuple[int, ...]:
    points = np.array(arr)
    return np.unravel_index(np.argmin(points), points.shape)


def generate_sho_config_minimum(
    interpolation: EnergyInterpolation, mass: float, initial_guess: float = 1.0
) -> Tuple[float, float]:
    points = np.array(interpolation["points"])
    min_coord = get_minimum_coordinate(points)
    min_z = min_coord[2]
    z_points = points[min_coord[0], min_coord[1]]
    z_indexes = np.arange(z_points.shape[0])

    far_edge_energy = z_points[-1]
    # We choose a region that is suitably harmonic
    # ie we cut off the tail of the potential
    fit_max_energy = 0.5 * far_edge_energy
    above_threshold = (z_indexes > min_z) & (z_points > fit_max_energy)
    # Stops at the first above threshold
    max_index: int = int(np.argmax(above_threshold) - 1)
    above_threshold = (z_indexes < min_z) & (z_points > fit_max_energy)

    # Search backwards, stops at the first above threshold
    min_index: int = z_points.shape[0] - np.argmax(above_threshold[::-1])

    z_offset = -interpolation["dz"] * min_z
    # Fit E = 1/2 * m * sho_omega ** 2 * z**2
    def fitting_f(z, sho_omega):
        return 0.5 * mass * (sho_omega * z) ** 2

    opt_params, _cov = scipy.optimize.curve_fit(
        f=fitting_f,
        xdata=np.arange(min_index, max_index + 1) * interpolation["dz"] + z_offset,
        ydata=z_points[min_index : max_index + 1],
        p0=[initial_guess],
    )
    # TODO: Change meaning of x_offset
    return opt_params[0], z_offset


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
