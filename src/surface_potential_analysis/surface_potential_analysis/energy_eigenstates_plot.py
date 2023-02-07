import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .energy_eigenstate import (
    EigenstateConfigUtil,
    EnergyEigenstates,
    filter_eigenstates_band,
)


def plot_eigenstate_positions(
    eigenstates: EnergyEigenstates, *, ax: Axes | None = None
) -> tuple[Figure, Axes, Line2D]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    (line,) = ax.plot(eigenstates["kx_points"], eigenstates["ky_points"])
    line.set_linestyle("None")
    line.set_marker("x")

    util = EigenstateConfigUtil(eigenstates["eigenstate_config"])
    dkx = np.abs(util.dkx1[0]) + np.abs(util.dkx2[0])
    ax.set_xlim(-(dkx) / 2, (dkx) / 2)
    dky = np.abs(util.dkx1[1]) + np.abs(util.dkx2[1])
    ax.set_ylim(-(dky) / 2, (dky) / 2)

    return fig, ax, line


def plot_nth_band_in_kx(
    eigenstates: EnergyEigenstates, n: int = 0, *, ax: Axes | None = None
):
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    filtered_eigenstates = filter_eigenstates_band(eigenstates, n=n)
    (line,) = ax.plot(
        filtered_eigenstates["kx_points"],
        filtered_eigenstates["eigenvalues"],
    )
    return fig, ax, line


def plot_lowest_band_in_kx(eigenstates: EnergyEigenstates, *, ax: Axes | None = None):
    return plot_nth_band_in_kx(eigenstates, ax=ax)
