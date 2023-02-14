from typing import Tuple

import numpy as np
from numpy.typing import NDArray


def get_point_fractions(shape: Tuple[int, int] = (8, 8), endpoint=True):
    """Get the coordinates as fractions of the momentum vectors"""
    x1_points = np.linspace(0, 1, shape[0], endpoint=endpoint)
    x2_points = np.linspace(0, 1, shape[1], endpoint=endpoint)

    x1v, x2v = np.meshgrid(x1_points, x2_points, indexing="ij")
    return np.array([x1v.ravel(), x2v.ravel()]).T


def grid_space(
    vec1: Tuple[float, float],
    vec2: Tuple[float, float],
    shape: Tuple[int, int],
    *,
    endpoint=True,
):
    """
    Layout points in a grid, with vec1, vec2 as the lattice vectors
    """
    fractions = get_point_fractions(shape, endpoint=endpoint)

    # Multiply the dk reciprocal lattuice vectors by their corresponding fraction
    # f1 * dk1 + f2 * dk2
    x_points = vec1[0] * fractions[:, 0] + vec2[0] * fractions[:, 1]
    y_points = vec1[1] * fractions[:, 0] + vec2[1] * fractions[:, 1]

    return np.array([x_points, y_points]).T


def get_coordinate_fractions(
    vec1: Tuple[float, float],
    vec2: Tuple[float, float],
    coordinates: NDArray,
):
    out = []
    print(vec1, vec2)
    for coord in coordinates:
        a = np.array(
            [
                [vec1[0], vec2[0]],
                [vec1[1], vec2[1]],
            ]
        )
        fraction = np.linalg.solve(a, [coord[0], coord[1]])
        out.append([fraction[0], fraction[1]])
    return np.array(out)


def get_points_in_brillouin_zone(
    dk1: Tuple[float, float],
    dk2: Tuple[float, float],
    *,
    size: Tuple[int, int] = (4, 4),
    include_zero=True,
):
    points = grid_space(dk1, dk2, shape=(2 * size[0], 2 * size[1]), endpoint=False)
    # Center points about k = 0
    # if not include_zero also offset by half the step size
    # note since endpoint is false 1 / 2 * size[0] is the fraction
    # per step in the x direction
    offset_fraction_x = 0.5 if include_zero else (0.5 - 1 / (4 * size[0]))
    offset_kx = offset_fraction_x * (dk1[0] + dk2[0])
    offset_fraction_y = 0.5 if include_zero else (0.5 - 1 / (4 * size[1]))
    offset_ky = offset_fraction_y * (dk1[1] + dk2[1])

    kx_points = points[:, 0] - offset_kx
    ky_points = points[:, 1] - offset_ky
    return np.array([kx_points, ky_points]).T
