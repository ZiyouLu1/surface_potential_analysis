import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .energy_eigenstate import EigenstateConfigUtil, EnergyEigenstates


def plot_eigenstate_positions(
    eigenstates: EnergyEigenstates, ax: Axes | None = None
) -> tuple[Figure, Axes, Line2D]:
    fig, ax1 = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    (line,) = ax1.plot(eigenstates["kx_points"], eigenstates["ky_points"])
    line.set_linestyle("None")
    line.set_marker("x")

    util = EigenstateConfigUtil(eigenstates["eigenstate_config"])
    dkx = np.abs(util.dkx1[0]) + np.abs(util.dkx2[0])
    ax1.set_xlim(-(dkx) / 2, (dkx) / 2)
    dky = np.abs(util.dkx1[1]) + np.abs(util.dkx2[1])
    ax1.set_ylim(-(dky) / 2, (dky) / 2)

    return fig, ax1, line


def plot_lowest_band_in_kx(eigenstates: EnergyEigenstates, ax: Axes | None = None):
    fig, a = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    kx_points = eigenstates["kx_points"][:6]
    eigenvalues = eigenstates["eigenvalues"][:6]

    (line,) = a.plot(kx_points, eigenvalues)
    return fig, a, line
