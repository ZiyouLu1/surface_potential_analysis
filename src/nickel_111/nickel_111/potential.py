import numpy as np

from surface_potential_analysis.energy_data import (
    EnergyGrid,
    EnergyPoints,
    load_energy_grid_legacy,
    load_energy_points,
)

from .surface_data import get_data_path


def load_raw_data() -> EnergyPoints:
    path = get_data_path("raw_data.json")
    return load_energy_points(path)


def generate_raw_unit_cell_data():
    data = load_raw_data()
    print(data["x_points"], data["y_points"])
    x_points = np.array(data["x_points"])
    y_points = np.array(data["y_points"])
    z_points = np.array(data["z_points"])

    x_c = np.sort(np.unique(x_points))
    y_c = np.sort(np.unique(y_points))
    z_c = np.sort(np.unique(z_points))
    points = np.array(data["points"])

    is_top = np.logical_and(x_points == x_c[0], y_points == y_c[0])
    top_points = [points[np.logical_and(is_top, z_points == z)][0] for z in z_c]

    is_top_fcc = np.logical_and(x_points == x_c[0], y_points == y_c[1])
    top_fcc_points = [points[np.logical_and(is_top_fcc, z_points == z)][0] for z in z_c]

    is_fcc = np.logical_and(x_points == x_c[0], y_points == y_c[2])
    fcc_points = [points[np.logical_and(is_fcc, z_points == z)][0] for z in z_c]

    is_hcp_top = np.logical_and(x_points == x_c[1], y_points == y_c[0])
    hcp_top_points = [points[np.logical_and(is_hcp_top, z_points == z)][0] for z in z_c]

    is_fcc_hcp = np.logical_and(x_points == x_c[1], y_points == y_c[2])
    fcc_hcp_points = [points[np.logical_and(is_fcc_hcp, z_points == z)][0] for z in z_c]

    is_hcp = np.logical_and(x_points == x_c[2], y_points == y_c[1])
    hcp_points = [points[np.logical_and(is_hcp, z_points == z)][0] for z in z_c]

    # Turns out we don't have enough points to produce an 'out' grid.

    print(fcc_points)


def load_john_interpolation() -> EnergyGrid:
    path = get_data_path("john_interpolated_data.json")
    return load_energy_grid_legacy(path)
