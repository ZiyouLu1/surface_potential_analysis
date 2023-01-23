from typing import Tuple

import numpy as np

from surface_potential_analysis.energy_eigenstate import (
    EigenstateConfig,
    EigenstateConfigUtil,
)


def get_point_fractions(grid_size=4, include_zero=True):

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

    x_points = dk1[0] * fractions[:, 0] + dk2[0] * fractions[:, 1]
    y_points = dk1[1] * fractions[:, 0] + dk2[1] * fractions[:, 1]

    out = np.array([x_points, y_points]).T
    np.testing.assert_array_equal(out[:, 0], x_points)
    np.testing.assert_array_equal(out[:, 1], y_points)
    return out


def get_brillouin_points_copper_100(
    config: EigenstateConfig, *, grid_size=4, include_zero=True
):
    util = EigenstateConfigUtil(config)
    dk1 = (util.dkx, 0)
    dk2 = (0, util.dky)
    return get_points_in_brillouin_zone(
        dk1, dk2, grid_size=grid_size, include_zero=include_zero
    )
