from typing import Any, Literal, TypeVar

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from surface_potential_analysis.basis import Basis

from .eigenstate_collection import EigenstateColllection

_BX0Inv = TypeVar("_BX0Inv", bound=Basis[Any, Any], covariant=True)
_BX1Inv = TypeVar("_BX1Inv", bound=Basis[Any, Any], covariant=True)
_BX2Inv = TypeVar("_BX2Inv", bound=Basis[Any, Any], covariant=True)


def get_projected_phases(
    phases: np.ndarray[tuple[int, Literal[3]], np.dtype[np.float_]],
    direction: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]],
) -> np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]:
    normalized_direction = direction / np.linalg.norm(direction)
    return np.dot(phases, normalized_direction)  # type: ignore


def plot_energies_against_bloch_phase_1D(
    collection: EigenstateColllection[_BX0Inv, _BX1Inv, _BX2Inv],
    direction: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]],
    band: int = 0,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    projected = get_projected_phases(collection["bloch_phases"], direction)
    (line,) = ax.plot(projected, collection["energies"][:, band])
    ax.set_xlabel("Bloch Phase")
    ax.set_ylabel("Energy / J")
    return fig, ax, line


def plot_lowest_band_energies_against_bloch_kx(
    collection: EigenstateColllection[_BX0Inv, _BX1Inv, _BX2Inv],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    return plot_energies_against_bloch_phase_1D(
        collection, np.array([1.0, 0, 0]), 0, ax=ax
    )
