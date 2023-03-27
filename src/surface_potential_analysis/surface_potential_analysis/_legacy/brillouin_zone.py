import numpy as np
from numpy.typing import NDArray


def get_point_fractions(shape: tuple[int, int] = (8, 8), endpoint=True):
    """Get the coordinates as fractions of the momentum vectors"""
    x0_points = np.linspace(0, 1, shape[0], endpoint=endpoint)
    x1_points = np.linspace(0, 1, shape[1], endpoint=endpoint)

    x0v, x1v = np.meshgrid(x0_points, x1_points, indexing="ij")
    return np.array([x0v.ravel(), x1v.ravel()]).T


def grid_space(
    vec0: tuple[float, float],
    vec1: tuple[float, float],
    shape: tuple[int, int],
    *,
    endpoint=True,
):
    """
    Layout points in a grid, with vec1, vec2 as the lattice vectors
    """
    fractions = get_point_fractions(shape, endpoint=endpoint)

    # Multiply the dk reciprocal lattuice vectors by their corresponding fraction
    # f1 * dk1 + f2 * dk2
    x_points = vec0[0] * fractions[:, 0] + vec1[0] * fractions[:, 1]
    y_points = vec0[1] * fractions[:, 0] + vec1[1] * fractions[:, 1]

    return np.array([x_points, y_points]).T


def get_points_in_brillouin_zone(
    dk0: tuple[float, float],
    dk1: tuple[float, float],
    *,
    size: tuple[int, int] = (4, 4),
    include_zero=True,
):
    points = grid_space(dk0, dk1, shape=(2 * size[0], 2 * size[1]), endpoint=False)
    # Center points about k = 0
    # if not include_zero also offset by half the step size
    # note since endpoint is false 1 / 2 * size[0] is the fraction
    # per step in the x direction
    offset_fraction_0 = 0.5 if include_zero else (0.5 - 1 / (4 * size[0]))
    offset_k0 = offset_fraction_0 * (dk0[0] + dk1[0])
    offset_fraction_1 = 0.5 if include_zero else (0.5 - 1 / (4 * size[1]))
    offset_k1 = offset_fraction_1 * (dk0[1] + dk1[1])

    k0_points = points[:, 0] - offset_k0
    k1_points = points[:, 1] - offset_k1
    return np.array([k0_points, k1_points]).T
