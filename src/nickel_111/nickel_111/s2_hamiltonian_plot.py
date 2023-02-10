import numpy as np

from surface_potential_analysis.energy_data import as_interpolation
from surface_potential_analysis.energy_eigenstate import EigenstateConfig
from surface_potential_analysis.sho_wavefunction_plot import (
    plot_energy_with_sho_potential_at_minimum,
)

from .s1_potential import load_john_interpolation
from .surface_data import save_figure


def plot_interpolation_with_sho_config() -> None:
    data = load_john_interpolation()
    interpolation = as_interpolation(data)
    config: EigenstateConfig = {
        "mass": 1.6735575e-27,
        "sho_omega": 198226131917441.6,  # 1.5e14,
        "delta_x0": data["delta_x0"],
        "delta_x1": data["delta_x1"],
        "resolution": (1, 1, 1),
    }

    fig, ax = plot_energy_with_sho_potential_at_minimum(
        interpolation, config, z_offset=interpolation["dz"]
    )
    ax.set_title("Plot of SHO config against Z")
    ax.legend()

    min_index = 17
    bottom_index = 40
    max_index = 63
    points = np.array(interpolation["points"])
    arg_min = np.unravel_index(np.argmin(points), points.shape)
    print(arg_min)
    z_idx = np.array([min_index, bottom_index, max_index])
    z_points = interpolation["dz"] * (z_idx - bottom_index)
    values = points[arg_min[0], arg_min[1], z_idx]
    (line,) = ax.plot(z_points, values)
    line.set_linestyle("")
    line.set_marker("x")

    fig.show()

    save_figure(fig, "sho_config_plot.png")

    input()
