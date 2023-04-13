from typing import Literal, TypeVar

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from surface_potential_analysis.util import get_measured_data

from .basis import (
    BasisUtil,
    ExplicitBasis,
    MomentumBasis,
    PositionBasis,
    explicit_momentum_basis_in_position,
)

_BX0Inv = TypeVar(
    "_BX0Inv", bound=ExplicitBasis[int, PositionBasis[int] | MomentumBasis[int]]
)


def plot_explicit_basis_states_x(
    basis: _BX0Inv,
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs", "angle"] = "abs",
) -> tuple[Figure, Axes, list[Line2D]]:
    """Plot basis states against position."""
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()
    basis_in_position: ExplicitBasis[int, PositionBasis[int]] = (
        basis  # type:ignore[assignment]
        if basis["parent"]["_type"] == "position"
        else explicit_momentum_basis_in_position(basis)  # type:ignore[arg-type]
    )
    util = BasisUtil(basis_in_position)

    x_points = np.linalg.norm(util.fundamental_x_points, axis=0)
    lines: list[Line2D] = []
    for i, vector in enumerate(basis_in_position["vectors"]):
        data = get_measured_data(vector, measure)
        (line,) = ax.plot(x_points, data)
        line.set_label(f"State {i}")
        lines.append(line)

    ax.set_xlabel("x / m")
    ax.set_ylabel("Amplitude")
    ax.set_title("Plot of the wavefunction of the explicit basis states")
    return fig, ax, lines


def plot_explicit_basis_state_x(
    basis: _BX0Inv,
    idx: int = 0,
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs", "angle"] = "abs",
) -> tuple[Figure, Axes, Line2D]:
    """Plot basis states against position."""
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()
    basis_in_position: ExplicitBasis[int, PositionBasis[int]] = (
        basis  # type:ignore[assignment]
        if basis["parent"]["_type"] == "position"
        else explicit_momentum_basis_in_position(basis)  # type:ignore[arg-type]
    )
    util = BasisUtil(basis_in_position)

    x_points = np.linalg.norm(util.fundamental_x_points, axis=0)
    data = get_measured_data(basis_in_position["vectors"][idx], measure)
    (line,) = ax.plot(x_points, data)
    return fig, ax, line
