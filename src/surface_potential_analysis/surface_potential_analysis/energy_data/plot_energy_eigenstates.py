import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .energy_eigenstates import EnergyEigenstates


def plot_eigenstate_positions(
    eigenstates: EnergyEigenstates, ax: Axes | None = None
) -> tuple[Figure, Axes, Line2D]:
    fig, ax1 = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    (line,) = ax1.plot(eigenstates["kx_points"], eigenstates["ky_points"])
    line.set_linestyle("None")
    line.set_marker("x")

    return fig, ax1, line


def plot_lowest_band_in_kx(
    eigenstates: EnergyEigenstates, ax: Axes | None = None
) -> tuple[Figure, Axes, Line2D]:
    fig, a = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    kx_points = eigenstates["kx_points"]
    eigenvalues = eigenstates["eigenvalues"]

    (line,) = a.plot(kx_points, eigenvalues)
    return fig, a, line
