from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from matplotlib import pyplot as plt

from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.operator.operator_list import (
    sum_diagonal_operator_list_over_axes,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from surface_potential_analysis.operator.operator_list import DiagonalOperatorList

    from .tunnelling_basis import TunnellingSimulationBasis

    _B0Inv = TypeVar("_B0Inv", bound=TunnellingSimulationBasis[Any, Any, Any])
    _L0Inv = TypeVar("_L0Inv", bound=int)


def plot_occupation_per_band(
    state: DiagonalOperatorList[_B0Inv, _B0Inv, _L0Inv],
    times: np.ndarray[tuple[_L0Inv], np.dtype[np.float_]],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot the occupation of each band in the simulation.

    Parameters
    ----------
    state : TunnellingSimulationState[_L0Inv, _N0Inv, _S0Inv]
    ax : Axes | None, optional
        plot axis, by default None

    Returns
    -------
    tuple[Figure, Axes]
    """
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    vectors_per_band = sum_diagonal_operator_list_over_axes(state, axes=(0, 1))
    for n in range(vectors_per_band["vectors"].shape[1]):
        (line,) = ax.plot(times, np.real_if_close(vectors_per_band["vectors"][:, n]))
        line.set_label(f"band {n}")

    ax.legend()
    ax.set_title("Plot of occupation of each band against time")
    ax.set_xlabel("time /s")
    ax.set_ylabel("occupation probability")
    return fig, ax


def plot_occupation_per_site(
    state: DiagonalOperatorList[_B0Inv, _B0Inv, _L0Inv],
    times: np.ndarray[tuple[_L0Inv], np.dtype[np.float_]],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot the occupation of each site in the system.

    Parameters
    ----------
    state : TunnellingSimulationState[_L0Inv, _S0Inv]
    ax : Axes | None, optional
        plot axis, by default None

    Returns
    -------
    tuple[Figure, Axes]
    """
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()
    vectors_per_band = sum_diagonal_operator_list_over_axes(state, axes=(2,))
    util = BasisUtil(vectors_per_band["basis"])

    for i in range(util.size):
        (line,) = ax.plot(times, np.real_if_close(vectors_per_band["vectors"][:, i]))
        nx0, nx1 = util.get_stacked_index(i)
        line.set_label(f"site ({nx0}, {nx1})")
    ax.legend()
    ax.set_title("Plot of occupation of each site against time")
    ax.set_xlabel("time /s")
    ax.set_ylabel("occupation probability")
    return fig, ax


def plot_occupation_per_state(
    state: DiagonalOperatorList[_B0Inv, _B0Inv, _L0Inv],
    times: np.ndarray[tuple[_L0Inv], np.dtype[np.float_]],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot the occupation of each state in the system.

    Parameters
    ----------
    state : TunnellingSimulationState[_L0Inv, _S0Inv]
    ax : Axes | None, optional
        plot axis, by default None

    Returns
    -------
    tuple[Figure, Axes]
    """
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()
    util = BasisUtil(state["basis"])

    for i, j, n in np.ndindex(*util.shape):
        ax.plot(times, np.real_if_close(state["vectors"][:, i, j, n]))

    ax.set_title("Plot of occupation of each state against time")
    ax.set_xlabel("time /s")
    ax.set_ylabel("occupation probability")
    return fig, ax
