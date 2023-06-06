from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import Boltzmann

from surface_potential_analysis.basis.util import BasisUtil

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from surface_potential_analysis.basis.basis import Basis
    from surface_potential_analysis.eigenstate.eigenstate_collection import (
        EigenstateColllection,
    )

    _B0Inv = TypeVar("_B0Inv", bound=Basis[Any])
    _L0Inv = TypeVar("_L0Inv", bound=int)
    _L1Inv = TypeVar("_L1Inv", bound=int)


def _get_projected_bloch_phases(
    collection: EigenstateColllection[_B0Inv, _L0Inv],
    direction: np.ndarray[tuple[_L1Inv], np.dtype[np.float_]],
) -> np.ndarray[tuple[_L0Inv], np.dtype[np.float_]]:
    util = BasisUtil(collection["basis"])
    bloch_phases = np.tensordot(
        collection["bloch_fractions"], util.fundamental_dk, axes=(1, 0)
    )
    normalized_direction = direction / np.linalg.norm(direction)
    return np.dot(bloch_phases, normalized_direction)  # type: ignore[no-any-return]


def plot_energies_against_bloch_phase_1d(
    collection: EigenstateColllection[_B0Inv, _L0Inv],
    direction: np.ndarray[tuple[int], np.dtype[np.float_]],
    band: int = 0,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the energies in an eigenstate collection against their projected phases.

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
    (line,) = ax.plot(projected, collection["energies"][:, band])
    ax.set_xlabel("Bloch Phase")
    ax.set_ylabel("Energy / J")
    return fig, ax, line


def plot_occupation_against_bloch_phase_1d(
    collection: EigenstateColllection[_B0Inv, _L0Inv],
    direction: np.ndarray[tuple[int], np.dtype[np.float_]],
    temperature: float,
    band: int = 0,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the energies in an eigenstate collection against their projected phases.

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
    energies = collection["energies"][:, band]
    occupations = np.exp(-energies / (temperature * Boltzmann))
    (line,) = ax.plot(projected, occupations)
    ax.set_xlabel("Bloch Phase / $m^{-1}$")
    ax.set_ylabel("Occupation / Au")
    return fig, ax, line


def plot_lowest_band_energies_against_bloch_k(
    collection: EigenstateColllection[_B0Inv, _L0Inv],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """
    plot energies against bloch phase in the k direction for the lowest band.

    Parameters
    ----------
    collection : EigenstateColllection[_B3d0Inv]
    ax : Axes | None, optional
        axis, by default None

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    direction = np.zeros(len(collection["basis"]))
    direction[0] = 1
    return plot_energies_against_bloch_phase_1d(collection, direction, 0, ax=ax)
