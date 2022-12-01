from typing import Tuple, TypedDict

import numpy as np
import scipy.optimize
from numpy.typing import ArrayLike

from energy_data import EnergyInterpolation


class SHOConfig(TypedDict):
    sho_omega: float
    """Angular frequency (in rad s-1) of the sho we will fit using"""
    mass: float
    """Mass in Kg"""
    z_offset: float
    """z position of the nz=0 position in the sho well"""


def get_minimum_coordinate(arr: ArrayLike) -> Tuple[int, ...]:
    points = np.array(arr)
    return np.unravel_index(np.argmin(points), points.shape)


def generate_sho_config_minimum(
    interpolation: EnergyInterpolation, mass: float
) -> SHOConfig:
    points = np.array(interpolation["points"])
    min_coord = get_minimum_coordinate(points)
    min_z = min_coord[2]
    z_points = points[min_coord[0], min_coord[1]]
    z_indexes = np.arange(z_points.shape[0])

    far_edge_energy = z_points[-1]
    # We choose a region that is suitably harmonic
    # ie we cut off the tail of the potential
    fit_max_energy = 0.8 * far_edge_energy
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
    )
    # TODO: Change meaning of x_offset
    return {"mass": mass, "sho_omega": opt_params[0], "z_offset": z_offset}
