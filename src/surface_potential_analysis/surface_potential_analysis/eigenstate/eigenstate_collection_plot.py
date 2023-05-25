from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from surface_potential_analysis.basis_config.basis_config import BasisConfig

    from .eigenstate_collection import EigenstateColllection

    _BC0Inv = TypeVar("_BC0Inv", bound=BasisConfig[Any, Any, Any])


def _get_projected_phases(
    phases: np.ndarray[tuple[int, Literal[3]], np.dtype[np.float_]],
    direction: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]],
) -> np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]:
    normalized_direction = direction / np.linalg.norm(direction)
    return np.dot(phases, normalized_direction)  # type: ignore[no-any-return]


def plot_energies_against_bloch_phase_1d(
    collection: EigenstateColllection[_BC0Inv],
    direction: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]],
    band: int = 0,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the energies in an eigenstate collection against their projected phases.

    Parameters
    ----------
    collection : EigenstateColllection[_BC0Inv]
    direction : np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    band : int, optional
        band to plot, by default 0
    ax : Axes | None, optional
        axis, by default None

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    projected = _get_projected_phases(collection["bloch_phases"], direction)
    (line,) = ax.plot(projected, collection["energies"][:, band])
    ax.set_xlabel("Bloch Phase")
    ax.set_ylabel("Energy / J")
    return fig, ax, line


def plot_lowest_band_energies_against_bloch_k(
    collection: EigenstateColllection[_BC0Inv],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """
    plot energies against bloch phase in the k direction for the lowest band.

    Parameters
    ----------
    collection : EigenstateColllection[_BC0Inv]
    ax : Axes | None, optional
        axis, by default None

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    return plot_energies_against_bloch_phase_1d(
        collection, np.array([1.0, 0, 0]), 0, ax=ax
    )
