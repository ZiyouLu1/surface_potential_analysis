from typing import Tuple

from surface_potential_analysis.energy_data import as_interpolation
from surface_potential_analysis.energy_eigenstate import (
    EigenstateConfig,
    generate_sho_config_minimum,
)
from surface_potential_analysis.hamiltonian import SurfaceHamiltonianUtil

from .potential import (
    load_interpolated_copper_data,
    load_spline_interpolated_relaxed_data,
)


def generate_hamiltonian(resolution: Tuple[int, int, int] = (1, 1, 1)):
    data = load_interpolated_copper_data()
    interpolation = as_interpolation(data)
    config: EigenstateConfig = {
        "mass": 1.6735575e-27,
        "sho_omega": 117905964225836.06,
        "delta_x1": data["delta_x1"],
        "delta_x2": data["delta_x2"],
        "resolution": resolution,
    }

    z_offset = -1.840551985155284e-10
    return SurfaceHamiltonianUtil(config, interpolation, z_offset)


def generate_hamiltonian_relaxed(resolution: Tuple[int, int, int] = (1, 1, 1)):
    data = load_spline_interpolated_relaxed_data()
    interpolation = as_interpolation(data)
    config: EigenstateConfig = {
        "mass": 1.6735575e-27,
        "sho_omega": 111119431700988.45,
        "delta_x1": data["delta_x1"],
        "delta_x2": data["delta_x2"],
        "resolution": resolution,
    }

    z_offset = -1.8866087481825024e-10
    return SurfaceHamiltonianUtil(config, interpolation, z_offset)


def generate_sho_config():
    data = load_spline_interpolated_relaxed_data()
    interpolation = as_interpolation(data)
    mass = 1.6735575e-27
    omega, z_offset = generate_sho_config_minimum(
        interpolation, mass, initial_guess=1e14
    )
    print(omega, z_offset)
