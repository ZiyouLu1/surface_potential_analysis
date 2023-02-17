from surface_potential_analysis.energy_data import as_interpolation
from surface_potential_analysis.energy_eigenstate import EigenstateConfig
from surface_potential_analysis.sho_wavefunction_plot import (
    plot_energy_with_sho_potential_at_minimum,
)

from .s1_potential import load_interpolated_grid


def plot_interpolation_with_sho_config() -> None:
    data = load_interpolated_grid()
    interpolation = as_interpolation(data)
    # 80% 99514067252307.23
    # 50% 117905964225836.06
    config: EigenstateConfig = {
        "mass": 1.6735575e-27,
        "sho_omega": 179704637926161.6,
        "delta_x0": data["delta_x0"],
        "delta_x1": data["delta_x1"],
        "resolution": (1, 1, 1),
    }
    z_offset = -9.848484848484871e-11

    fig, _ = plot_energy_with_sho_potential_at_minimum(interpolation, config, z_offset)
    fig.show()
    input()
