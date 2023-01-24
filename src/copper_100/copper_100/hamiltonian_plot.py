from surface_potential_analysis.energy_data import as_interpolation, get_xy_points_delta
from surface_potential_analysis.energy_eigenstate import EigenstateConfig
from surface_potential_analysis.sho_wavefunction_plot import (
    plot_energy_with_sho_potential_at_hollow,
)
from surface_potential_analysis.surface_hamiltonian_plot import plot_nth_eigenstate

from .hamiltonian import generate_hamiltonian
from .potential import load_interpolated_copper_data


def plot_interpolation_with_sho_config() -> None:
    data = load_interpolated_copper_data()
    interpolation = as_interpolation(data)
    # 80% 99514067252307.23
    # 50% 117905964225836.06
    config: EigenstateConfig = {
        "mass": 1.6735575e-27,
        "sho_omega": 117905964225836.06,  # 1e14,
        "delta_x1": (get_xy_points_delta(data["x_points"]), 0),
        "delta_x2": (0, get_xy_points_delta(data["y_points"])),
        "resolution": (1, 1, 1),
    }
    z_offset = -1.840551985155284e-10
    plot_energy_with_sho_potential_at_hollow(interpolation, config, z_offset)


def plot_copper_ground_eigenvector():
    h = generate_hamiltonian(resolution=(12, 12, 10))
    fig, _ = plot_nth_eigenstate(h)

    fig.show()
    input()
