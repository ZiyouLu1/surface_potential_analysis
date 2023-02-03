from typing import Dict, List, Tuple

import numpy as np

from surface_potential_analysis.energy_data import (
    EnergyGrid,
    EnergyPoints,
    interpolate_energy_grid_fourier,
    load_energy_grid,
    load_energy_grid_legacy,
    load_energy_points,
    normalize_energy,
    save_energy_grid,
    truncate_energy,
)

from .surface_data import get_data_path


def load_raw_data() -> EnergyPoints:
    path = get_data_path("raw_data.json")
    return load_energy_points(path)


def load_raw_data_grid() -> EnergyGrid:
    path = get_data_path("raw_data_reflected.json")
    return load_energy_grid(path)


def load_cleaned_data_grid() -> EnergyGrid:
    data = load_raw_data_grid()
    normalized = normalize_energy(data)  # 1.7e-19
    return truncate_energy(normalized, cutoff=0.5e-19, n=6, offset=1e-20)


def load_john_interpolation() -> EnergyGrid:
    path = get_data_path("john_interpolated_data.json")
    return load_energy_grid_legacy(path)


def load_interpolated_grid() -> EnergyGrid:
    path = get_data_path("interpolated_data.json")
    return load_energy_grid(path)


# def reflect_coordinate(
#     coord: Tuple[float, float], perpendicular_line: Tuple[float, float]
# ):
#     coord_a = np.array(coord)
#     perpendicular_line_a = np.divide(
#         perpendicular_line, np.linalg.norm(perpendicular_line)
#     )

#     return tuple(
#         coord_a - 2 * np.dot(coord_a, perpendicular_line_a) * perpendicular_line_a
#     )


# def fold_coordinate_into_lhp(
#     delta_x1: Tuple[float, float],
#     delta_x2: Tuple[float, float],
#     coord: Tuple[float, float],
# ):
#     v_symmetry_perp = (-delta_x2[0], delta_x1[1])
#     is_lhp = np.dot(coord, np.divide(delta_x1, np.linalg.norm(delta_x1))) > np.dot(
#         coord, np.divide(delta_x2, np.linalg.norm(delta_x2))
#     )
#     if is_lhp:
#         print("is_lhp")
#         return coord
#     print("pass")
#     return reflect_coordinate(coord, v_symmetry_perp)


# def get_coordinate_in_irreducible_region(
#     delta_x1: Tuple[float, float],
#     delta_x2: Tuple[float, float],
#     coord: Tuple[float, float],
# ):
#     return fold_coordinate_into_lhp(delta_x1, delta_x2, coord)


def map_irreducible_points_into_unit_cell(
    irreducible_points: EnergyPoints,
) -> EnergyGrid:

    z_points = np.sort(np.unique(irreducible_points["z_points"]))
    xy_points = np.unique(
        np.array([irreducible_points["x_points"], irreducible_points["y_points"]]).T,
        axis=0,
    )

    mapping_to_xy = {
        (8, -4): (11, 4),
        (6, -3): (9, 3),
        (7, -3): (10, 4),
        (4, -2): (6, 2),
        (5, -2): (7, 3),
        (6, -2): (8, 4),
        (7, -2): (9, 5),
        (2, -1): (3, 1),
        (3, -1): (4, 2),
        (4, -1): (5, 3),
        (5, -1): (6, 4),
        (6, -1): (7, 5),
        (0, +0): (0, 0),
        (1, +0): (1, 1),
        (2, +0): (2, 2),
        (3, +0): (3, 3),
        (4, +0): (4, 4),
        (5, +0): (5, 5),
        (6, +0): (6, 6),
        (1, +1): (0, 2),
        (2, +1): (1, 3),
        (3, +1): (2, 4),
        (4, +1): (3, 5),
        (5, +1): (4, 6),
        (2, +2): (0, 4),
        (3, +2): (1, 5),
        (4, +2): (2, 6),
        (5, +2): (3, 7),
        (3, +3): (0, 6),
        (4, +3): (1, 7),
        (4, +4): (0, 8),
    }

    # mapping = [
    #     [(8,-4), (7,-3), (6,-2), (5,-1), (4, 0), (3, 1), (2, 2), (3, 1), (4, 0), (5,-1), (6,-2), (7,-3), (8,-4)],
    #     [(7,-3), (6,-3), (5,-2), (4,-1), (3, 0), (2, 1), (2, 1), (3, 0), (4,-1), (5,-2), (6,-3), (7,-3), (7,-3)],
    #     [(6,-2), (5,-2), (4,-2), (3,-1), (2, 0), (1, 1), (2, 0), (3,-1), (4,-2), (5,-2), (6,-2), (7,-2), (6,-2)],
    #     [(5,-1), (4,-1), (3,-1), (2,-1), (1, 0), (1, 0), (2,-1), (3,-1), (4,-1), (5,-1), (6,-1), (6,-1), (5,-1)],
    #     [(4,+0), (3, 0), (2, 0), (1, 0), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (5, 0), (4,+0)],
    #     [(3,+1), (2,+1), (1,+1), (1,+0), (1, 0), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (5, 1), (4, 1), (3,+1)],
    #     [(2,+2), (2,+1), (2,+0), (2,-1), (2, 0), (2, 1), (2, 2), (3, 2), (4, 2), (5, 2), (4, 2), (3, 2), (2,+2)],
    #     [(3,+1), (3,+0), (3,-1), (3,-1), (3, 0), (3, 1), (3, 2), (3, 3), (4, 3), (4, 3), (3, 3), (3, 2), (3,+1)],
    #     [(4,+0), (4,-1), (4,-2), (4,-1), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 3), (4, 2), (4, 1), (4,+0)],
    #     [(5,-1), (5,-2), (5,-2), (5,-1), (5, 0), (5, 1), (5, 2), (4, 3), (4, 3), (5, 2), (5, 1), (5, 0), (5,-1)],
    #     [(6,-2), (6,-3), (6,-2), (6,-1), (6, 0), (5, 1), (4, 2), (3, 3), (4, 2), (5, 1), (6, 0), (6,-1), (6,-2)],
    #     [(7,-3), (7,-3), (7,-2), (6,-1), (5, 0), (4, 1), (3, 2), (3, 2), (4, 1), (5, 0), (6,-1), (7,-2), (7,-3)],
    #     [(8,-4), (7,-3), (6,-2), (5,-1), (4, 0), (3, 1), (2, 2), (3, 1), (4, 0), (5,-1), (6,-2), (7,-3), (8,-4)],
    # ]
    mapping = [
        [
            (8, -4),
            (7, -3),
            (6, -2),
            (5, -1),
            (4, 0),
            (3, 1),
            (2, 2),
            (3, 1),
            (4, 0),
            (5, -1),
            (6, -2),
            (7, -3),
        ],
        [
            (7, -3),
            (6, -3),
            (5, -2),
            (4, -1),
            (3, 0),
            (2, 1),
            (2, 1),
            (3, 0),
            (4, -1),
            (5, -2),
            (6, -3),
            (7, -3),
        ],
        [
            (6, -2),
            (5, -2),
            (4, -2),
            (3, -1),
            (2, 0),
            (1, 1),
            (2, 0),
            (3, -1),
            (4, -2),
            (5, -2),
            (6, -2),
            (7, -2),
        ],
        [
            (5, -1),
            (4, -1),
            (3, -1),
            (2, -1),
            (1, 0),
            (1, 0),
            (2, -1),
            (3, -1),
            (4, -1),
            (5, -1),
            (6, -1),
            (6, -1),
        ],
        [
            (4, +0),
            (3, 0),
            (2, 0),
            (1, 0),
            (0, 0),
            (1, 0),
            (2, 0),
            (3, 0),
            (4, 0),
            (5, 0),
            (6, 0),
            (5, 0),
        ],
        [
            (3, +1),
            (2, +1),
            (1, +1),
            (1, +0),
            (1, 0),
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 1),
            (5, 1),
            (5, 1),
            (4, 1),
        ],
        [
            (2, +2),
            (2, +1),
            (2, +0),
            (2, -1),
            (2, 0),
            (2, 1),
            (2, 2),
            (3, 2),
            (4, 2),
            (5, 2),
            (4, 2),
            (3, 2),
        ],
        [
            (3, +1),
            (3, +0),
            (3, -1),
            (3, -1),
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 3),
            (4, 3),
            (4, 3),
            (3, 3),
            (3, 2),
        ],
        [
            (4, +0),
            (4, -1),
            (4, -2),
            (4, -1),
            (4, 0),
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 4),
            (4, 3),
            (4, 2),
            (4, 1),
        ],
        [
            (5, -1),
            (5, -2),
            (5, -2),
            (5, -1),
            (5, 0),
            (5, 1),
            (5, 2),
            (4, 3),
            (4, 3),
            (5, 2),
            (5, 1),
            (5, 0),
        ],
        [
            (6, -2),
            (6, -3),
            (6, -2),
            (6, -1),
            (6, 0),
            (5, 1),
            (4, 2),
            (3, 3),
            (4, 2),
            (5, 1),
            (6, 0),
            (6, -1),
        ],
        [
            (7, -3),
            (7, -3),
            (7, -2),
            (6, -1),
            (5, 0),
            (4, 1),
            (3, 2),
            (3, 2),
            (4, 1),
            (5, 0),
            (6, -1),
            (7, -2),
        ],
    ]
    mapping_in_xy = [[mapping_to_xy[k] for k in m] for m in mapping]

    z_points_in_xy: Dict[Tuple[int, int], List[float]] = {}

    for z in z_points:
        for (x, y) in xy_points:

            ix = np.argwhere(x == np.sort(np.unique(irreducible_points["x_points"])))[
                0
            ][0]
            iy = np.argwhere(y == np.sort(np.unique(irreducible_points["y_points"])))[
                0
            ][0]

            curr = z_points_in_xy.get((ix, iy), [])

            isAtXYZ = np.logical_and(
                np.logical_and(
                    np.array(irreducible_points["x_points"]) == x,
                    np.array(irreducible_points["y_points"]) == y,
                ),
                np.array(irreducible_points["z_points"]) == z,
            )
            if np.count_nonzero(isAtXYZ) != 1:
                raise AssertionError("More than one point found")
            curr.append(np.array(irreducible_points["points"])[isAtXYZ][0])
            z_points_in_xy[(ix, iy)] = curr

    final_grid = [[z_points_in_xy[xy_coord] for xy_coord in m] for m in mapping_in_xy]

    x_width = np.max(irreducible_points["x_points"]) - np.min(
        irreducible_points["x_points"]
    )
    y_height = np.max(irreducible_points["y_points"]) - np.min(
        irreducible_points["y_points"]
    )
    diagonal_length = np.sqrt(np.square(y_height / 2) + np.square(x_width))

    delta_x1 = (2 * x_width, 0)
    delta_x2 = (0.5 * delta_x1[0], np.sqrt(3) * delta_x1[0] / 2)

    if not np.allclose(delta_x2[1], y_height / 2 + diagonal_length):
        raise AssertionError(
            f"{delta_x2[1]} not close to {y_height / 2 + diagonal_length}"
        )
    return {
        "delta_x1": delta_x1,
        "delta_x2": delta_x2,
        "points": final_grid,
        "z_points": z_points.tolist(),
    }


def generate_reflected_data():
    irreducible_points = load_raw_data()
    data = map_irreducible_points_into_unit_cell(irreducible_points)
    path = get_data_path("raw_data_reflected.json")
    save_energy_grid(data, path)


def generate_interpolated_data():
    grid = load_cleaned_data_grid()

    data = interpolate_energy_grid_fourier(grid, (40, 40, 100))
    path = get_data_path("interpolated_data.json")
    save_energy_grid(data, path)
