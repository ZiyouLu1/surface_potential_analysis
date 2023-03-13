from surface_potential_analysis.energy_data import as_interpolation
from surface_potential_analysis.energy_eigenstate import EigenstateConfig
from surface_potential_analysis.sho_wavefunction_plot import (
    plot_energy_with_sho_potential_at_hollow,
)

from .s1_potential import (
    load_interpolated_copper_data,
    load_spline_interpolated_relaxed_data,
)


def plot_interpolation_with_sho_config() -> None:
    data = load_interpolated_copper_data()
    interpolation = as_interpolation(data)
    # 80% 99514067252307.23
    # 50% 117905964225836.06
    config: EigenstateConfig = {
        "mass": 1.6735575e-27,
        "sho_omega": 117905964225836.06,  # 1e14,
        "delta_x0": data["delta_x0"],
        "delta_x1": data["delta_x1"],
        "resolution": (1, 1, 1),
    }
    z_offset = -1.840551985155284e-10
    plot_energy_with_sho_potential_at_hollow(interpolation, config, z_offset)


def plot_relaxed_interpolation_with_sho_config() -> None:
    data = load_spline_interpolated_relaxed_data()
    interpolation = as_interpolation(data)
    # 80% 99514067252307.23
    # 50% 117905964225836.06
    config: EigenstateConfig = {
        "mass": 1.6735575e-27,
        "sho_omega": 111119431700988.45,  # 1e14,
        "delta_x0": data["delta_x0"],
        "delta_x1": data["delta_x1"],
        "resolution": (1, 1, 1),
    }
    z_offset = -1.8866087481825024e-10
    fig, _ = plot_energy_with_sho_potential_at_hollow(interpolation, config, z_offset)
    fig.show()
    input()
