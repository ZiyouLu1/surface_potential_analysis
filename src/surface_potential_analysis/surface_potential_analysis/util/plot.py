from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation
from matplotlib.colors import Normalize, SymLogNorm

from surface_potential_analysis.stacked_basis.util import (
    get_x_coordinates_in_axes,
)

from .util import Measure, get_data_in_axes, get_measured_data

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.figure import Figure
    from matplotlib.image import AxesImage

    from surface_potential_analysis.axis.stacked_axis import StackedBasisLike
    from surface_potential_analysis.types import SingleStackedIndexLike


Scale = Literal["symlog", "linear"]


def get_norm_with_clim(
    scale: Scale,
    clim: tuple[float | None, float | None] = (None, None),
) -> Normalize:
    """
    Get the appropriate norm given the scale and clim.

    Parameters
    ----------
    scale : Scale
    clim : tuple[float  |  None, float  |  None], optional

    Returns
    -------
    Normalize
    """
    match scale:
        case "linear":
            return Normalize(vmin=clim[0], vmax=clim[1])
        case "symlog":
            abs_0 = -1 if clim[0] is None else np.abs(clim[0])
            abs_1 = -1 if clim[1] is None else np.abs(clim[1])
            max_abs = max([abs_0, abs_1])
            return SymLogNorm(
                vmin=clim[0],
                vmax=clim[1],
                linthresh=1 if max([abs_0, abs_1]) <= 0 else np.abs(1e-4 * max_abs),  # type: ignore No parameter named "linthresh"
            )


def build_animation(
    build_frame: Callable[[int, Axes], QuadMesh | AxesImage],
    n: int,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    clim: tuple[float | None, float | None] = (None, None),
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    Build an animation from the data, set the scale and clim to the correct values.

    Parameters
    ----------
    build_frame : Callable[[int, Axes], QuadMesh | AxesImage]
        function to generate each frame
    n : int
        number of frames to generate
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"
    clim : tuple[float  |  None, float  |  None], optional
        plot clim, by default (None, None)

    Returns
    -------
    tuple[Figure, Axes, ArtistAnimation]
    """
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    mesh0 = build_frame(0, ax)
    frames = [[build_frame(d, ax)] for d in range(n)]

    c_max: float = (
        float(np.max([i[0].get_clim()[1] for i in frames]))  # type: ignore np.min returns float
        if clim[1] is None
        else clim[1]
    )
    c_min: float = (
        float(np.min([i[0].get_clim()[0] for i in frames]))  # type: ignore np.min returns float
        if clim[0] is None
        else clim[0]
    )
    norm = get_norm_with_clim(scale, (c_min, c_max))
    for (mesh,) in frames:
        mesh.set_norm(norm)
        mesh.set_clim(c_min, c_max)
    mesh0.set_norm(norm)
    mesh0.set_clim(c_min, c_max)

    return (fig, ax, ArtistAnimation(fig, frames))


_L0Inv = TypeVar("_L0Inv", bound=int)


# ruff: noqa: PLR0913


def animate_through_surface_x(
    basis: StackedBasisLike[*tuple[Any, ...]],
    points: np.ndarray[tuple[_L0Inv], np.dtype[np.complex_ | np.float_ | np.bool_]],
    axes: tuple[int, int, int] = (0, 1, 2),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    clim: tuple[float | None, float | None] = (None, None),
    measure: Measure = "abs",
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    Given data on a given coordinate grid in 3D, animate through z_axis.

    Parameters
    ----------
    coordinates : np.ndarray[tuple[Literal[3], Unpack[_S0Inv]], np.dtype[np.float_]]
    data : np.ndarray[_S0Inv, np.dtype[np.float_]]
    z_axis : Literal[0, 1, 2, -1, -2, -3]
        axis through which to animate
    ax : Axes | None, optional
        plot axis, by default None
    scale : Scale, optional
        scale, by default "linear"
    clim : tuple[float  |  None, float  |  None], optional
        clim, by default (None, None)

    Returns
    -------
    tuple[Figure, Axes, ArtistAnimation]
    """
    idx = tuple(0 for _ in range(basis.ndim - 3)) if idx is None else idx
    clim = (0.0, clim[1]) if clim[0] is None and measure == "abs" else clim

    coordinates = get_x_coordinates_in_axes(basis, axes, idx)
    data = get_measured_data(get_data_in_axes(points, axes, idx), measure)

    fig, ax, ani = build_animation(
        lambda i, ax: ax.pcolormesh(
            *coordinates[:, :, :, i],
            data[:, :, i],
            shading="nearest",
        ),
        data.shape[2],
        ax=ax,
        scale=scale,
        clim=clim,
    )
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(ax.collections[0], ax=ax, format="%4.1e")  # type: ignore Type of "collections" is unknown

    ax.set_xlabel(f"x{axes[0]} axis")
    ax.set_ylabel(f"x{axes[1]} axis")
    return fig, ax, ani
