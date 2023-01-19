from typing import Tuple

from surface_potential_analysis.energy_data import as_interpolation, get_xy_points_delta
from surface_potential_analysis.energy_eigenstate import (
    EigenstateConfig,
    generate_sho_config_minimum,
)
from surface_potential_analysis.hamiltonian import SurfaceHamiltonianUtil

from .potential import load_interpolated_copper_data


def generate_hamiltonian(resolution: Tuple[int, int, int] = (1, 1, 1)):
    data = load_interpolated_copper_data()
    interpolation = as_interpolation(data)
    config: EigenstateConfig = {
        "mass": 1.6735575e-27,
        "sho_omega": 117905964225836.06,
        "delta_x": get_xy_points_delta(data["x_points"]),
        "delta_y": get_xy_points_delta(data["y_points"]),
        "resolution": resolution,
    }

    z_offset = -1.840551985155284e-10
    return SurfaceHamiltonianUtil(config, interpolation, z_offset)


def generate_sho_config():
    data = load_interpolated_copper_data()
    interpolation = as_interpolation(data)
    mass = 1.6735575e-27
    return generate_sho_config_minimum(interpolation, mass, initial_guess=1e14)
