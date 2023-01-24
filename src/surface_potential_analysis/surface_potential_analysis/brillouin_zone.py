from typing import Tuple

import numpy as np

from surface_potential_analysis.energy_eigenstate import (
    EigenstateConfig,
    EigenstateConfigUtil,
)


def get_point_fractions(grid_size=4, include_zero=True):
    """Get the coordinates as fractions of the momentum vectors"""
    (kx_points, kx_step) = np.linspace(
        -0.5, 0.5, 2 * grid_size, endpoint=False, retstep=True
    )
    (ky_points, ky_step) = np.linspace(
        -0.5, 0.5, 2 * grid_size, endpoint=False, retstep=True
    )
    if not include_zero:
        kx_points += kx_step / 2
        ky_points += ky_step / 2

    xv, yv = np.meshgrid(kx_points, ky_points)
    return np.array([xv.ravel(), yv.ravel()]).T


def get_points_in_brillouin_zone(
    dk1: Tuple[float, float],
    dk2: Tuple[float, float],
    *,
    grid_size=4,
    include_zero=True,
):
    fractions = get_point_fractions(grid_size, include_zero)

    # Multiply the dk reciprocal lattuice vectors by their corresponding fraction
    # f1 * dk1 + f2 * dk2
    x_points = dk1[0] * fractions[:, 0] + dk2[0] * fractions[:, 1]
    y_points = dk1[1] * fractions[:, 0] + dk2[1] * fractions[:, 1]

    return np.array([x_points, y_points]).T


def get_brillouin_points_irreducible_config(
    config: EigenstateConfig, *, grid_size=4, include_zero=True
):
    """
    If the eigenstate config is that of the irreducible unit cell
    we can use the dkx of the lattuice to generate the brillouin zone points
    """
    util = EigenstateConfigUtil(config)
    return get_points_in_brillouin_zone(
        util.dkx1, util.dkx2, grid_size=grid_size, include_zero=include_zero
    )
