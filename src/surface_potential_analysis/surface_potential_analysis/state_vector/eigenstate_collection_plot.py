from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import Boltzmann

from surface_potential_analysis.axis.util import BasisUtil
from surface_potential_analysis.state_vector.state_vector_list import (
    state_vector_list_into_iter,
)

from .plot import plot_state_1d_x

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from surface_potential_analysis.axis.axis_like import BasisLike
    from surface_potential_analysis.axis.block_fraction_axis import (
        AxisWithBlockFractionLike,
    )
    from surface_potential_analysis.axis.stacked_axis import (
        StackedBasisLike,
    )
    from surface_potential_analysis.state_vector.eigenstate_collection import (
        EigenstateColllection,
    )
    from surface_potential_analysis.state_vector.state_vector_list import (
        StateVectorList,
    )
    from surface_potential_analysis.types import SingleStackedIndexLike
    from surface_potential_analysis.util.plot import Scale
    from surface_potential_analysis.util.util import Measure

    _B0 = TypeVar("_B0", bound=BasisLike[Any, Any])
    _SB0 = TypeVar("_SB0", bound=StackedBasisLike[*tuple[Any, ...]])
    _L0 = TypeVar("_L0", bound=int)
    _BF0 = TypeVar("_BF0", bound=AxisWithBlockFractionLike[Any, Any])


# ruff: noqa: PLR0913


def _get_projected_bloch_phases(
    collection: EigenstateColllection[StackedBasisLike[_BF0, Any], Any],
    direction: np.ndarray[tuple[_L0], np.dtype[np.float_]],
) -> np.ndarray[tuple[int], np.dtype[np.float_]]:
    util = BasisUtil(collection["basis"][1])
    bloch_phases = np.tensordot(
        collection["basis"][0][0].bloch_fractions,
        util.fundamental_dk_stacked,
        axes=(1, 0),
    )
    normalized_direction = direction / np.linalg.norm(direction)
    return np.dot(bloch_phases, normalized_direction)


def plot_states_1d_x(
    states: StateVectorList[_B0, _SB0],
    axes: tuple[int] = (0,),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes]:
    """
    Plot all states in a StateVectorList.

    Parameters
    ----------
    states : StateVectorList[_B0Inv, _L0Inv]
    axis : int, optional
        axis to plot along, by default 0
    idx : SingleStackedIndexLike | None, optional
        index in axes perpendicular to axis, by default None
    ax : Axes | None, optional
        plot axis, by default None
    measure : Measure, optional
        measure, by default "abs"
    scale : Scale, optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes]
    """
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    for state in state_vector_list_into_iter(states):
        plot_state_1d_x(state, axes, idx, ax=ax, measure=measure, scale=scale)
    return fig, ax


def plot_eigenvalues_against_bloch_phase_1d(
    collection: EigenstateColllection[StackedBasisLike[_BF0, Any], Any],
    direction: np.ndarray[tuple[int], np.dtype[np.float_]],
    band: int = 0,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the eigenvalues in an eigenstate collection against their projected phases.

    Parameters
    ----------
    collection : EigenstateColllection[_B0Inv, _L0Inv]
    direction : np.ndarray[tuple[int], np.dtype[np.float_]]
    band : int, optional
        band to plot, by default 0
    ax : Axes | None, optional
        axis, by default None

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    projected = _get_projected_bloch_phases(collection, direction)
    (line,) = ax.plot(
        projected,
        collection["eigenvalue"].reshape(*collection["basis"][0].shape)[:, band],
    )
    ax.set_xlabel("Bloch Phase")
    ax.set_ylabel("Energy / J")
    return fig, ax, line


def plot_occupation_against_bloch_phase_1d(
    collection: EigenstateColllection[StackedBasisLike[_BF0, Any], Any],
    direction: np.ndarray[tuple[int], np.dtype[np.float_]],
    temperature: float,
    band: int = 0,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the eigenvalues in an eigenstate collection against their projected phases.

    Parameters
    ----------
    collection : EigenstateColllection[_B0Inv, _L0Inv]
    direction : np.ndarray[tuple[int], np.dtype[np.float_]]
    band : int, optional
        band to plot, by default 0
    ax : Axes | None, optional
        axis, by default None

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    projected = _get_projected_bloch_phases(collection, direction)
    eigenvalues = collection["eigenvalue"].reshape(*collection["basis"][0].shape, -1)[
        :, band
    ]
    occupations = np.exp(-eigenvalues / (temperature * Boltzmann))
    (line,) = ax.plot(projected, occupations)
    ax.set_xlabel("Bloch Phase / $m^{-1}$")
    ax.set_ylabel("Occupation / Au")
    return fig, ax, line


def plot_lowest_band_eigenvalues_against_bloch_k(
    collection: EigenstateColllection[StackedBasisLike[_BF0, Any], Any],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """
    plot eigenvalues against bloch phase in the k direction for the lowest band.

    Parameters
    ----------
    collection : EigenstateColllection[_B3d0Inv]
    ax : Axes | None, optional
        axis, by default None

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    direction = np.zeros(collection["basis"].n)
    direction[0] = 1
    return plot_eigenvalues_against_bloch_phase_1d(collection, direction, 0, ax=ax)
