from __future__ import annotations

import numpy as np
from surface_potential_analysis.basis.util import AxisWithLengthBasisUtil
from surface_potential_analysis.potential.plot import (
    plot_potential_1d_x2_comparison_111,
    plot_potential_2d_x,
)
from surface_potential_analysis.potential.plot_point_potential import (
    get_point_potential_xy_locations,
    plot_point_potential_all_z,
    plot_point_potential_location_xy,
)
from surface_potential_analysis.potential.plot_uneven_potential import (
    plot_uneven_potential_z_comparison_111,
)
from surface_potential_analysis.potential.potential import normalize_potential

from .s1_potential import (
    get_interpolated_potential,
    get_reflected_potential,
    load_raw_data,
)
from .surface_data import save_figure


def plot_raw_data_points() -> None:
    data = load_raw_data()
    fig, ax, _ = plot_point_potential_location_xy(data)

    locations = get_point_potential_xy_locations(data)
    e_min: list[float] = []
    for x, y in locations.T:
        idx = np.argwhere(
            np.logical_and(
                np.array(data["x_points"]) == x,
                np.array(data["y_points"]) == y,
            )
        )
        e_min.append(np.min(np.array(data["points"])[idx]))

    amin = np.argsort(e_min)
    x_min = locations[0][amin[0]]
    y_min = locations[1][amin[0]]
    ax.text(x_min, y_min, "FCC (lowest E)")

    x_min = locations[0][amin[1]]
    y_min = locations[1][amin[1]]
    ax.text(x_min, y_min, "HCP (second lowest E)")

    for i in range(2, 9):
        x_min = locations[0][amin[i]]
        y_min = locations[1][amin[i]]
        ax.text(x_min, y_min, f"({i})")

    x_min = locations[0][amin[-1]]
    y_min = locations[1][amin[-1]]
    ax.text(x_min, y_min, "Top (largest E)")

    fig.show()
    save_figure(fig, "raw_points.png")

    fig, ax = plot_point_potential_all_z(data)
    ax.set_ylim(-4.66e-17, -4.59e-17)

    ax.legend()
    fig.show()
    save_figure(fig, "raw_points_z.png")

    input()


def plot_interpolated_potential_2d() -> None:
    potential = get_interpolated_potential((50, 50, 100))
    util = AxisWithLengthBasisUtil(potential["basis"])
    min_x2 = util.get_stacked_index(np.argmin(potential["vector"]))[2]

    fig, _, _ = plot_potential_2d_x(potential, (0, 1), (min_x2,), scale="symlog")
    fig.show()
    input()


def plot_interpolated_potential_comparison() -> None:
    potential = get_interpolated_potential((50, 50, 100))

    raw_data = normalize_potential(get_reflected_potential())

    fig, ax, _ = plot_potential_1d_x2_comparison_111(potential)
    _, _, lines = plot_uneven_potential_z_comparison_111(raw_data, ax=ax)
    for ln in lines:
        ln.set_marker("x")
        ln.set_linestyle("")
    ax.set_ylim(bottom=0, top=1e-18)
    fig.show()
    input()
