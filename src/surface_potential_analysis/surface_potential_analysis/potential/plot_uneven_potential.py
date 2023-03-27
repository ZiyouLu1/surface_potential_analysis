from typing import Literal, TypeVar

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from surface_potential_analysis.potential.potential import UnevenPotential

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)


def plot_uneven_potential_z(
    potential: UnevenPotential[_L0Inv, _L1Inv, _L2Inv],
    idx: tuple[int, int],
    *,
    ax: Axes | None = None,
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, Line2D]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    coordinates = potential["basis"][2]
    data = potential["points"][*idx, :]

    (line,) = ax.plot(coordinates, data)
    ax.set_xlabel("Z axis")
    ax.set_ylabel("Energy /J")
    ax.set_yscale(scale)
    return fig, ax, line


def plot_uneven_potential_z_comparison(
    potential: UnevenPotential[_L0Inv, _L1Inv, _L2Inv],
    comparison_points: dict[str, tuple[int, int]],
    *,
    ax: Axes | None = None,
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()
    for label, idx in comparison_points.items():
        (_, _, line) = plot_uneven_potential_z(potential, idx, ax=ax, scale=scale)
        line.set_label(label)
    ax.legend()
    return fig, ax
