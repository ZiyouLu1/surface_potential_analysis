from typing import Literal

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from surface_potential_analysis.util import get_measured_data

from .basis import BasisUtil, ExplicitBasis, PositionBasis


def plot_explicit_basis_states(
    basis: ExplicitBasis[int, PositionBasis[int]],
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs", "angle"] = "abs",
) -> tuple[Figure, Axes]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()
    util = BasisUtil(basis)

    x_points = np.linalg.norm(util.fundamental_x_points, axis=0)
    for i, vector in enumerate(basis["vectors"]):
        data = get_measured_data(vector, measure)
        (line,) = ax.plot(x_points, data)
        line.set_label(f"State {i}")

    ax.set_xlabel("x / m")
    ax.set_ylabel("Amplitude")
    ax.set_title("Plot of the wavefunction of the explicit basis states")
    return fig, ax


def plot_explicit_basis_state(
    basis: ExplicitBasis[int, PositionBasis[int]],
    idx: int = 0,
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs", "angle"] = "abs",
) -> tuple[Figure, Axes, Line2D]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()
    util = BasisUtil(basis)

    x_points = np.linalg.norm(util.fundamental_x_points, axis=0)
    data = get_measured_data(basis["vectors"][0], measure)
    (line,) = ax.plot(x_points, data)
    return fig, ax, line
