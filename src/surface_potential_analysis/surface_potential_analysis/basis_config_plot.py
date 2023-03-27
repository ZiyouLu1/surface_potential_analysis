from typing import Literal, TypeVar

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from surface_potential_analysis.basis_config import PositionBasisConfig
from surface_potential_analysis.eigenstate.eigenstate_plot import get_projected_x_points
from surface_potential_analysis.util import slice_along_axis

_L0Inv = TypeVar("_L0", bound=int)
_L1 = TypeVar("_L1", bound=int)
_L2Inv = TypeVar("_L2", bound=int)


def plot_projected_coordinates_2D(
    basis: PositionBasisConfig[_L0Inv, _L1, _L2Inv],
    idx: int,
    z_axis: Literal[0, 1, 2, -1, -2, -3],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()
    coordinates = get_projected_x_points(basis, z_axis)[
        slice_along_axis(idx, (z_axis % 3) + 1)
    ]

    (line,) = ax.plot(*coordinates)
    line.set_linestyle(" ")
    line.set_marker("x")

    ax.set_xlabel(f"x{0 if (z_axis % 3) != 0 else 1} axis")
    ax.set_ylabel(f"x{2 if (z_axis % 3) != 2 else 1} axis")
    ax.set_aspect("equal", adjustable="box")

    return fig, ax, line
