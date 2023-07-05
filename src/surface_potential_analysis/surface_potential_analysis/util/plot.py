from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation
from matplotlib.colors import Normalize, SymLogNorm

from surface_potential_analysis.util.util import slice_along_axis

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.figure import Figure
    from matplotlib.image import AxesImage


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
            return SymLogNorm(
                vmin=clim[0],
                vmax=clim[1],
                linthresh=None if clim[1] is None else 1e-4 * clim[1],
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

    frames: list[list[QuadMesh | AxesImage]] = []
    for d in range(n):
        frames.append([build_frame(d, ax)])

    c_max: float = (
        np.max([i[0].get_clim()[1] for i in frames]) if clim[1] is None else clim[1]
    )
    c_min: float = (
        np.min([i[0].get_clim()[0] for i in frames]) if clim[0] is None else clim[0]
    )
    norm = get_norm_with_clim(scale, (c_min, c_max))
    for (mesh,) in frames:
        mesh.set_norm(norm)
        mesh.set_clim(c_min, c_max)
    mesh0.set_norm(norm)
    mesh0.set_clim(c_min, c_max)

    return (fig, ax, ArtistAnimation(fig, frames))


_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)


def animate_through_surface(
    coordinates: np.ndarray[
        tuple[Literal[2], _L0Inv, _L1Inv, _L2Inv], np.dtype[np.float_]
    ],
    data: np.ndarray[tuple[_L0Inv, _L1Inv, _L2Inv], np.dtype[np.float_]],
    axes: tuple[int, int],
    z_axis: Literal[0, 1, 2, -1, -2, -3],
    idx: SingleStackedIndexLike,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    clim: tuple[float | None, float | None] = (None, None),
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
    fig, ax, ani = build_animation(
        lambda i, ax: ax.pcolormesh(
            *coordinates[slice_along_axis(i, (z_axis % 3) + 1)],
            data[slice_along_axis(i, (z_axis % 3))],
            shading="nearest",
        ),
        data.shape[z_axis],
        ax=ax,
        scale=scale,
        clim=clim,
    )
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(ax.collections[0], ax=ax, format="%4.1e")

    ax.set_xlabel(f"x{0 if (z_axis % 3) != 0 else 1} axis")
    ax.set_ylabel(f"x{2 if (z_axis % 3) != 2 else 1} axis")  # noqa: PLR2004
    return fig, ax, ani
