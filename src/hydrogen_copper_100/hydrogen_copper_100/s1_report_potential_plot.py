from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from surface_potential_analysis.basis.plot import (
    plot_fundamental_x_at_index_projected_2d,
    plot_fundamental_x_in_plane_projected_2d,
)
from surface_potential_analysis.basis.util import AxisWithLengthBasisUtil
from surface_potential_analysis.potential.plot import (
    plot_potential_1d_x2_comparison_100,
)
from surface_potential_analysis.potential.plot_uneven_potential import (
    plot_uneven_potential_z_comparison_100,
)
from surface_potential_analysis.potential.potential import (
    UnevenPotential3dZAxis,
    normalize_potential,
)

from .s1_potential import (
    get_interpolated_potential,
    get_interpolated_potential_relaxed,
    load_raw_copper_potential,
)

PLOT_COLOURS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def plot_raw_potential_points() -> None:
    fig, ax = plt.subplots()
    potential = load_raw_copper_potential()
    xy_basis = (potential["basis"][0], potential["basis"][1])
    util = AxisWithLengthBasisUtil(xy_basis)

    idx = (
        np.array([0, 1, 2, 3, 4, 1, 2, 3, 4, 2, 3, 4, 3, 4, 4]),
        np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4]),
    )
    _, _, line = plot_fundamental_x_at_index_projected_2d(xy_basis, idx, (0, 1), ax=ax)
    line.set_marker("X")
    line.set_color(PLOT_COLOURS[1])

    _, _, line = plot_fundamental_x_in_plane_projected_2d(xy_basis, (0, 1), (), ax=ax)
    line.set_color(PLOT_COLOURS[0])

    ax.set_xlim([-0.5 * util.dx[0, 0], 9 * util.dx[0, 0]])
    ax.set_ylim([-0.5 * util.dx[1, 1], 9 * util.dx[1, 1]])
    fig.show()

    input()


def plot_copper_potential_interpolation() -> None:
    fig, ax = plt.subplots()

    potential = load_raw_copper_potential()
    potential = normalize_potential(potential)
    potential["basis"] = (
        potential["basis"][0],
        potential["basis"][1],
        UnevenPotential3dZAxis(
            potential["basis"][2].z_points - potential["basis"][2].z_points[0]
        ),
    )
    _, _, lines = plot_uneven_potential_z_comparison_100(potential, ax=ax)
    for i, ln in enumerate(lines):
        ln.set_marker("x")
        ln.set_linestyle("")
        ln.set_color(PLOT_COLOURS[i])

    interpolated = get_interpolated_potential((100, 100, 100))
    _, _, lines = plot_potential_1d_x2_comparison_100(interpolated, ax=ax)
    for i, ln in enumerate(lines):
        ln.set_color(PLOT_COLOURS[i])

    lines = []
    for i in range(3):
        (line,) = ax.plot([], [])
        line.set_linestyle("-")
        line.set_marker("x")
        line.set_color(PLOT_COLOURS[i])
        lines.append(line)
    ax.legend(lines, ["Top Site", "Bridge Site", "Hollow Site"])

    ax.set_ylim(bottom=0, top=1e-18)
    fig.show()
    input()


def plot_copper_relaxed_potential_comparison() -> None:
    fig, ax = plt.subplots()

    interpolated = get_interpolated_potential((100, 100, 100))
    _, _, lines = plot_potential_1d_x2_comparison_100(interpolated, ax=ax)
    for i, ln in enumerate(lines):
        ln.set_color(PLOT_COLOURS[i])

    interpolated = get_interpolated_potential_relaxed((100, 100, 100))
    _, _, lines = plot_potential_1d_x2_comparison_100(interpolated, ax=ax)
    for i, ln in enumerate(lines):
        ln.set_color(PLOT_COLOURS[i])
        ln.set_linestyle("--")

    lines = []
    for i in range(3):
        (line,) = ax.plot([], [])
        line.set_linestyle("-")
        line.set_color(PLOT_COLOURS[i])
        lines.append(line)
    ax.legend(lines, ["Top Site", "Bridge Site", "Hollow Site"])

    ax.set_ylim(bottom=0, top=1e-18)
    fig.show()
    input()
