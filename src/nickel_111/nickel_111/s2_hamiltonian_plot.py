import numpy as np

from .s1_potential import load_interpolated_grid
from .surface_data import save_figure


def plot_interpolation_with_sho_config() -> None:
    data = load_interpolated_grid()
    interpolation = as_interpolation(data)
    config: EigenstateConfig = {
        "mass": 1.6735575e-27,
        "sho_omega": 195636899474736.66,  # 1.5e14,
        "delta_x0": data["delta_x0"],
        "delta_x1": data["delta_x1"],
        "resolution": (1, 1, 1),
    }

    fig, ax = plot_energy_with_sho_potential_at_minimum(
        interpolation, config, z_offset=-1.0000000000000004e-10
    )
    ax.set_title("Plot of SHO config against Z")
    ax.legend()

    min_index = 25
    bottom_index = 45
    max_index = 66
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
