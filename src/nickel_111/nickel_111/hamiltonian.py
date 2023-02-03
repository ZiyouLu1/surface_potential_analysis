from typing import Tuple

from surface_potential_analysis.energy_data import as_interpolation
from surface_potential_analysis.energy_eigenstate import (
    EigenstateConfig,
    generate_sho_config_minimum,
)
from surface_potential_analysis.hamiltonian import SurfaceHamiltonianUtil

from .potential import load_john_interpolation


def generate_hamiltonian(resolution: Tuple[int, int, int] = (1, 1, 1)):
    data = load_john_interpolation()
    interpolation = as_interpolation(data)
    config: EigenstateConfig = {
        "mass": 1.6735575e-27,
        "sho_omega": 198226131917441.6,
        "delta_x1": data["delta_x1"],
        "delta_x2": data["delta_x2"],
        "resolution": resolution,
    }

    z_offset = -1.0000000000000017e-10
    return SurfaceHamiltonianUtil(config, interpolation, z_offset)


def generate_sho_config():
    data = load_john_interpolation()
    interpolation = as_interpolation(data)
    mass = 1.6735575e-27
    omega, z_offset = generate_sho_config_minimum(
        interpolation, mass, initial_guess=1.5e14, fit_max_energy_fraction=0.3
    )
    print(omega, z_offset)
