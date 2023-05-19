from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
from matplotlib import pyplot as plt
from surface_potential_analysis.basis_config.basis_config import (
    BasisConfig,
    BasisConfigUtil,
    PositionBasisConfigUtil,
    get_fundamental_projected_x_points,
)
from surface_potential_analysis.util.plot import (
    Scale,
    animate_through_surface,
)
from surface_potential_analysis.wavepacket.eigenstate_conversion import (
    get_unfurled_basis,
)

if TYPE_CHECKING:
    from matplotlib.animation import ArtistAnimation
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from surface_potential_analysis.basis.basis import PositionBasis

    from surface_dynamics_simulation.tunnelling_simulation.tunnelling_simulation_state import (
        TunnellingSimulationState,
    )

_L0Inv = TypeVar("_L0Inv", bound=int)
_S0Inv = TypeVar("_S0Inv", bound=tuple[int, int, int])


def plot_occupation_per_band(
    state: TunnellingSimulationState[_L0Inv, _S0Inv], *, ax: Axes | None = None
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

    vectors_per_band = np.sum(
        state["vectors"].reshape(*state["shape"], -1), axis=(0, 1)
    )
    for n in range(state["shape"][2]):
        (line,) = ax.plot(state["times"], vectors_per_band[n])
        line.set_label(f"band {n}")

    ax.legend()
    ax.set_title("Plot of occupation of each band against time")
    ax.set_xlabel("time /s")
    ax.set_ylabel("occupation probability")
    return fig, ax


def plot_occupation_per_site(
    state: TunnellingSimulationState[_L0Inv, _S0Inv], *, ax: Axes | None = None
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
    vectors_per_site = np.sum(state["vectors"].reshape(*state["shape"], -1), axis=(2))
    for i, j in np.ndindex(*state["shape"][0:2]):
        (line,) = ax.plot(state["times"], vectors_per_site[i, j])
        line.set_label(f"site ({i}, {j})")
    ax.legend()
    ax.set_title("Plot of occupation of each site against time")
    ax.set_xlabel("time /s")
    ax.set_ylabel("occupation probability")
    return fig, ax


if TYPE_CHECKING:
    _BC0Inv = TypeVar("_BC0Inv", bound=BasisConfig[Any, Any, Any])
    _PBS = PositionBasis[Literal[1]]
    _SPB = BasisConfig[_PBS, _PBS, _PBS]


def _get_single_point_basis(basis: _BC0Inv) -> _SPB:
    fundamental = BasisConfigUtil(basis).get_fundamental_basis_in("position")
    return (
        {"_type": "position", "delta_x": fundamental[0]["delta_x"], "n": 1},
        {"_type": "position", "delta_x": fundamental[1]["delta_x"], "n": 1},
        {"_type": "position", "delta_x": fundamental[2]["delta_x"], "n": 1},
    )


def animate_occupation_per_site_2d(
    state: TunnellingSimulationState[_L0Inv, _S0Inv],
    basis: _BC0Inv | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    Plot the occupation of each site in the system in a 2D grid.

    Parameters
    ----------
    state : TunnellingSimulationState[_L0Inv, _S0Inv]
    ax : Axis | None, optional
        plot axis, by default None

    Returns
    -------
    tuple[Figure, Axis, ArtistAnimation]
    """
    shape = state["shape"]
    unfurled_basis = get_unfurled_basis(
        _get_single_point_basis(basis)
        if basis is not None
        else PositionBasisConfigUtil.from_resolution((1, 1, 1)),
        (shape[0], shape[1]),
    )

    occupations = np.real(np.sum(state["vectors"].reshape(*shape, -1), axis=(2)))
    x_coordinates = get_fundamental_projected_x_points(unfurled_basis, 2)
    coordinates = np.repeat(x_coordinates, occupations.shape[2], axis=3)

    fig, ax, ani = animate_through_surface(
        coordinates, occupations, 2, ax=ax, scale=scale, clim=(0, 1)
    )

    ax.set_title("Plot of occupation of each site against time")
    return fig, ax, ani


def plot_occupation_per_state(
    state: TunnellingSimulationState[_L0Inv, _S0Inv], *, ax: Axes | None = None
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
    vectors_per_state = state["vectors"].reshape(*state["shape"], -1)
    for i, j, n in np.ndindex(*state["shape"][0:3]):
        ax.plot(state["times"], vectors_per_state[i, j, n])

    ax.set_title("Plot of occupation of each state against time")
    ax.set_xlabel("time /s")
    ax.set_ylabel("occupation probability")
    return fig, ax


def plot_flux_from_site(
    state: TunnellingSimulationState[_L0Inv, _S0Inv], *, ax: Axes | None = None
) -> tuple[Figure, Axes]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    ax.set_xlabel("time /s")
    ax.set_ylabel("flux from site")
    return fig, ax
