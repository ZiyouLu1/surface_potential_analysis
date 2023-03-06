from surface_potential_analysis.energy_data import as_interpolation
from surface_potential_analysis.energy_eigenstate import (
    EigenstateConfig,
    generate_sho_config_minimum,
)
from surface_potential_analysis.hamiltonian import SurfaceHamiltonianUtil

from .s1_potential import load_interpolated_grid


def generate_hamiltonian(resolution: tuple[int, int, int] = (1, 1, 1)):
    data = load_interpolated_grid()
    interpolation = as_interpolation(data)
    config: EigenstateConfig = {
        "mass": 1.6735575e-27,
        "sho_omega": 179704637926161.6,
        "delta_x0": data["delta_x0"],
        "delta_x1": data["delta_x1"],
        "resolution": resolution,
    }

    z_offset = -9.848484848484871e-11
    return SurfaceHamiltonianUtil(config, interpolation, z_offset)


def generate_sho_config():
    data = load_interpolated_grid()
    interpolation = as_interpolation(data)
    mass = 1.6735575e-27
    omega, z_offset = generate_sho_config_minimum(
        interpolation, mass, initial_guess=1e14, fit_max_energy_fraction=0.3
    )
    print(omega, z_offset)
