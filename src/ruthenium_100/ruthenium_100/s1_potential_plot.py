import numpy as np

from surface_potential_analysis.energy_data import get_energy_points_xy_locations
from surface_potential_analysis.energy_data_plot import (
    plot_all_energy_points_z,
    plot_energy_points_location,
)

from .s1_potential import load_raw_data
from .surface_data import save_figure


def plot_raw_data_points():
    data = load_raw_data()
    fig, ax, _ = plot_energy_points_location(data)

    locations = get_energy_points_xy_locations(data)
    e_min = []
    for (x, y) in locations:
        idx = np.argwhere(
            np.logical_and(
                np.array(data["x_points"]) == x,
                np.array(data["y_points"]) == y,
            )
        )
        e_min.append(np.min(np.array(data["points"])[idx]))

    amin = np.argsort(e_min)
    x_min = locations[amin[0]][0]
    y_min = locations[amin[0]][1]
    ax.text(x_min, y_min, "FCC (lowest E)")

    x_min = locations[amin[1]][0]
    y_min = locations[amin[1]][1]
    ax.text(x_min, y_min, "HCP (second lowest E)")

    for i in range(2, 9):
        x_min = locations[amin[i]][0]
        y_min = locations[amin[i]][1]
        ax.text(x_min, y_min, f"({i})")

    x_min = locations[amin[-1]][0]
    y_min = locations[amin[-1]][1]
    ax.text(x_min, y_min, "Top (largest E)")

    fig.show()
    save_figure(fig, "raw_points.png")

    fig, ax = plot_all_energy_points_z(data)
    # ax.set_ylim(0, 3 * 10**-19)

    ax.legend()
    fig.show()
    save_figure(fig, "raw_points_z.png")

    input()
