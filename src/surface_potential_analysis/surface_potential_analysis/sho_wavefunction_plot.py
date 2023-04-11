import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy.constants import hbar

from surface_potential_analysis.sho_wavefunction import calculate_sho_wavefunction

from .energy_data import EnergyInterpolation
from .energy_eigenstate import EigenstateConfig


def plot_energy_with_sho_potential(
    interpolation: EnergyInterpolation,
    eigenstate_config: EigenstateConfig,
    z_offset: float,
    xy_ind: tuple[int, int],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    points = np.array(interpolation["points"])
    start_z = z_offset
    end_z = interpolation["dz"] * (points.shape[2] - 1) + z_offset
    z_points = np.linspace(start_z, end_z, points.shape[2])

    (line1,) = ax.plot(z_points, points[xy_ind[0], xy_ind[1]])
    line1.set_label("Potential at center point")

    sho_pot = (
        0.5
        * eigenstate_config["mass"]
        * (eigenstate_config["sho_omega"] * z_points) ** 2
    )
    (line2,) = ax.plot(z_points, sho_pot)
    line2.set_label("SHO Config")
    ax.legend()
    ax.set_title(
        "Plot of the potential superimposed by the SHO potential\n"
        "used to generate eigenstates"
    )
    ax.set_xlabel("Position /m")
    ax.set_ylabel("Energy / J")

    return fig, ax


def plot_energy_with_sho_potential_at_hollow(
    interpolation: EnergyInterpolation,
    eigenstate_config: EigenstateConfig,
    z_offset: float,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    points = np.array(interpolation["points"])
    middle_x_index = math.floor(points.shape[0] / 2)
    middle_y_index = math.floor(points.shape[1] / 2)
    xy_ind = (middle_x_index, middle_y_index)

    fig, ax = plot_energy_with_sho_potential(
        interpolation, eigenstate_config, z_offset, xy_ind, ax=ax
    )

    max_potential = 1e-18
    ax.set_ylim(0, max_potential)
    return fig, ax


def plot_energy_with_sho_potential_at_minimum(
    interpolation: EnergyInterpolation,
    eigenstate_config: EigenstateConfig,
    z_offset: float,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    points = np.array(interpolation["points"], dtype=float)
    arg_min = np.unravel_index(np.argmin(points), points.shape)
    xy_ind = (int(arg_min[0]), int(arg_min[1]))

    fig, ax = plot_energy_with_sho_potential(
        interpolation, eigenstate_config, z_offset, xy_ind, ax=ax
    )

    max_potential = 1e-18
    ax.set_ylim(0, max_potential)
    return fig, ax


def plot_sho_wavefunctions(
    z_points: list[float],
    sho_omega: float,
    mass: float,
    first_n: int = 3,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, list[Line2D]]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    lines: list[Line2D] = []
    for n in range(first_n):
        wfn = calculate_sho_wavefunction(z_points, sho_omega, mass, n)
        wfn *= (0.25 * hbar * sho_omega) / np.max(wfn)
        wfn += (hbar * sho_omega) * (n + 0.5)
        (ln,) = ax.plot(z_points, wfn)
        ln.set_label(f"Sho N={n}")
        lines.append(ln)

    return fig, ax, lines