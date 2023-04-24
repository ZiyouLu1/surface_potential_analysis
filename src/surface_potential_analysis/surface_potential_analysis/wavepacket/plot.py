from typing import Any, Literal, TypeVar

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from surface_potential_analysis.basis.basis import Basis, MomentumBasis
from surface_potential_analysis.basis_config.basis_config import (
    BasisConfig,
    BasisConfigUtil,
    get_fundamental_projected_k_points,
    get_fundamental_projected_x_points,
)
from surface_potential_analysis.eigenstate.eigenstate import (
    convert_eigenstate_to_position_basis,
)
from surface_potential_analysis.eigenstate.plot import (
    animate_eigenstate_3d,
    plot_eigenstate_2d,
    plot_eigenstate_along_path,
    plot_eigenstate_difference_2d,
)
from surface_potential_analysis.util import get_measured_data
from surface_potential_analysis.wavepacket.eigenstate_conversion import (
    unfurl_wavepacket,
)

from .wavepacket import (
    MomentumBasisWavepacket,
    Wavepacket,
    get_wavepacket_sample_frequencies,
)

_NS0Inv = TypeVar("_NS0Inv", bound=int)
_NS1Inv = TypeVar("_NS1Inv", bound=int)

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)

_BC0Inv = TypeVar("_BC0Inv", bound=BasisConfig[Any, Any, Any])

_BX0Inv = TypeVar("_BX0Inv", bound=Basis[Any, Any])
_BX1Inv = TypeVar("_BX1Inv", bound=Basis[Any, Any])
_BX2Inv = TypeVar("_BX2Inv", bound=Basis[Any, Any])


def plot_wavepacket_sample_frequencies(
    wavepacket: Wavepacket[_NS0Inv, _NS1Inv, _BC0Inv],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the frequencies used to sample the wavepacket.

    Parameters
    ----------
    wavepacket : Wavepacket[_NS0Inv, _NS1Inv, _BC0Inv]
    ax : Axes | None, optional
        plot axis, by default None

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    fractions = get_wavepacket_sample_frequencies(
        wavepacket["basis"], np.array(wavepacket["vectors"].shape)[0:2]
    )[:2, :]
    (line,) = ax.plot(*fractions.reshape(2, -1))
    line.set_marker("x")
    line.set_linestyle("")

    ax.set_xlabel("kx /$m^{-1}$")
    ax.set_ylabel("ky /$m^{-1}$")
    ax.set_title("Plot of sample points in the wavepacket")

    return fig, ax, line


def get_wavepacket_sample_basis(
    wavepacket: Wavepacket[_NS0Inv, _NS1Inv, BasisConfig[_BX0Inv, _BX1Inv, _BX2Inv]]
) -> BasisConfig[
    MomentumBasis[_NS0Inv], MomentumBasis[_NS1Inv], MomentumBasis[Literal[1]]
]:
    """
    Get the basis used to sample the brillouin zone.

    Parameters
    ----------
    wavepacket : Wavepacket[_NS0Inv, _NS1, BasisConfig[_BX0Inv, _BX1Inv, _BX2Inv]]

    Returns
    -------
    BasisConfig[MomentumBasis[_NS0Inv], MomentumBasis[_NS1Inv], MomentumBasis[Literal[1]]]
    """
    (ns0, ns1) = wavepacket["energies"].shape
    util = BasisConfigUtil(wavepacket["basis"])
    return (
        {
            "_type": "momentum",
            "delta_x": util.delta_x0 * ns0,
            "n": ns0,
        },
        {
            "_type": "momentum",
            "delta_x": util.delta_x1 * ns1,
            "n": ns1,
        },
        {
            "_type": "momentum",
            "delta_x": util.delta_x2,
            "n": 1,
        },
    )


def plot_wavepacket_energies_momentum(
    wavepacket: Wavepacket[_NS0Inv, _NS1Inv, BasisConfig[_BX0Inv, _BX1Inv, _BX2Inv]],
    *,
    ax: Axes | None = None,
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot the energy of the eigenstates in a wavepacket.

    Parameters
    ----------
    wavepacket : Wavepacket[_NS0Inv, _NS1Inv, BasisConfig[_BX0Inv, _BX1Inv, _BX2Inv]]
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
    """
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    basis = get_wavepacket_sample_basis(wavepacket)
    coordinates = get_fundamental_projected_k_points(basis, 2)[:, :, :, 0]
    points = np.fft.ifftshift(wavepacket["energies"])

    shifted_coordinates = np.fft.ifftshift(coordinates, axes=(1, 2))

    mesh = ax.pcolormesh(*shifted_coordinates, points, shading="nearest")
    mesh.set_norm(scale)
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(mesh, ax=ax, format="%4.1e")

    ax.set_xlabel("kx axis")
    ax.set_ylabel("ky axis")
    ax.set_title("Plot of the band energies against momentum")

    return fig, ax, mesh


def plot_wavepacket_energies_position(
    wavepacket: Wavepacket[_NS0Inv, _NS1Inv, BasisConfig[_BX0Inv, _BX1Inv, _BX2Inv]],
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot the fourier transform of energy of the eigenstates in a wavepacket.

    Parameters
    ----------
    wavepacket : Wavepacket[_NS0Inv, _NS1Inv, BasisConfig[_BX0Inv, _BX1Inv, _BX2Inv]]
    ax : Axes | None, optional
        plot axis, by default None
    measure : Literal[&quot;real&quot;, &quot;imag&quot;, &quot;abs&quot;], optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
    """
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    basis = get_wavepacket_sample_basis(wavepacket)
    coordinates = get_fundamental_projected_x_points(basis, 2)[:, :, :, 0]

    data = np.fft.ifft2(wavepacket["energies"])
    data[0, 0] = 0
    points = get_measured_data(data, measure)

    mesh = ax.pcolormesh(*coordinates, points, shading="nearest")
    mesh.set_norm(scale)
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(mesh, ax=ax, format="%4.1e")

    ax.set_xlabel("x axis")
    ax.set_ylabel("y axis")
    ax.set_title("Plot of the fourier transform of the band energies against position")

    return fig, ax, mesh


def plot_wavepacket_2d(
    wavepacket: MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv],
    idx: int,
    z_axis: Literal[0, 1, 2, -1, -2, -3],
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot wavepacket in 2D at idx along the given axis.

    Parameters
    ----------
    wavepacket : MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv]
        Wavepacket in momentum basis
    idx : int
        index along z_axis
    z_axis : Literal[0, 1, 2]
        z_axis, perpendicular to plotted direction
    ax : Axes | None, optional
        plot axis, by default None
    measure : Literal[&quot;real&quot;, &quot;imag&quot;, &quot;abs&quot;], optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
    """
    eigenstate_momentum = unfurl_wavepacket(wavepacket)
    eigenstate = convert_eigenstate_to_position_basis(eigenstate_momentum)
    return plot_eigenstate_2d(
        eigenstate, idx, z_axis, ax=ax, measure=measure, scale=scale
    )


def plot_wavepacket_x0x1(
    wavepacket: MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv],
    x2_idx: int,
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot wavepacket perpendicular to the x2 axis.

    Parameters
    ----------
    wavepacket : MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv]
        Wavepacket in momentum basis
    x2_idx : int
        index along x2 axis
    ax : Axes | None, optional
        plot axis, by default None
    measure : Literal[&quot;real&quot;, &quot;imag&quot;, &quot;abs&quot;], optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
    """
    return plot_wavepacket_2d(
        wavepacket, x2_idx, 2, ax=ax, measure=measure, scale=scale
    )


def plot_wavepacket_x1x2(
    wavepacket: MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv],
    x0_idx: int,
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot wavepacket perpendicular to the x0 axis.

    Parameters
    ----------
    wavepacket : MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv]
        Wavepacket in momentum basis
    x0_idx : int
        index along x0 axis
    ax : Axes | None, optional
        plot axis, by default None
    measure : Literal[&quot;real&quot;, &quot;imag&quot;, &quot;abs&quot;], optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
    """
    return plot_wavepacket_2d(
        wavepacket, x0_idx, 0, ax=ax, measure=measure, scale=scale
    )


def plot_wavepacket_x2x0(
    wavepacket: MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv],
    x1_idx: int,
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot wavepacket perpendicular to the x1 axis.

    Parameters
    ----------
    wavepacket : MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv]
        Wavepacket in momentum basis
    x1_idx : int
        index along x1 axis
    ax : Axes | None, optional
        plot axis, by default None
    measure : Literal[&quot;real&quot;, &quot;imag&quot;, &quot;abs&quot;], optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
    """
    return plot_wavepacket_2d(
        wavepacket, x1_idx, 1, ax=ax, measure=measure, scale=scale
    )


def plot_wavepacket_difference_2d(
    wavepacket0: MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv],
    wavepacket1: MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv],
    idx: int,
    z_axis: Literal[0, 1, 2, -1, -2, -3],
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot the difference between two wavepackets in 2D.

    Parameters
    ----------
    wavepacket0 : MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv]
    wavepacket1 : MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv]
    idx : int
        index along z_axis to plot
    z_axis : Literal[0, 1, 2,
        direction perpendicular to which to plot
    ax : Axes | None, optional
        plot axis, by default None
    measure : Literal[&quot;real&quot;, &quot;imag&quot;, &quot;abs&quot;], optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
    """
    eigenstate_momentum0 = unfurl_wavepacket(wavepacket0)
    eigenstate0 = convert_eigenstate_to_position_basis(eigenstate_momentum0)

    eigenstate_momentum1 = unfurl_wavepacket(wavepacket1)
    eigenstate1 = convert_eigenstate_to_position_basis(eigenstate_momentum1)

    return plot_eigenstate_difference_2d(
        eigenstate0, eigenstate1, idx, z_axis, ax=ax, measure=measure, scale=scale
    )


def animate_wavepacket_3d(
    wavepacket: MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv],
    z_axis: Literal[0, 1, 2, -1, -2, -3],
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    Animate the wavepacket in 3D, perpendicular to z_axis.

    Parameters
    ----------
    wavepacket : MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv]
    z_axis : Literal[0, 1, 2,
        direction along which to animate
    ax : Axes | None, optional
        plot axis, by default None
    measure : Literal[&quot;real&quot;, &quot;imag&quot;, &quot;abs&quot;], optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, ArtistAnimation]
    """
    eigenstate_momentum = unfurl_wavepacket(wavepacket)
    eigenstate = convert_eigenstate_to_position_basis(eigenstate_momentum)
    return animate_eigenstate_3d(
        eigenstate, z_axis, ax=ax, measure=measure, scale=scale
    )


def animate_wavepacket_x0x1(
    wavepacket: MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv],
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    Animate the wavepacket in 3D, perpendicular to x2.

    Parameters
    ----------
    wavepacket : MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv]
    ax : Axes | None, optional
        plot axis, by default None
    measure : Literal[&quot;real&quot;, &quot;imag&quot;, &quot;abs&quot;], optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, ArtistAnimation]
    """
    return animate_wavepacket_3d(wavepacket, 2, ax=ax, measure=measure, scale=scale)


def animate_wavepacket_x1x2(
    wavepacket: MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv],
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    Animate the wavepacket in 3D, perpendicular to x0.

    Parameters
    ----------
    wavepacket : MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv]
    ax : Axes | None, optional
        plot axis, by default None
    measure : Literal[&quot;real&quot;, &quot;imag&quot;, &quot;abs&quot;], optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, ArtistAnimation]
    """
    return animate_wavepacket_3d(wavepacket, 0, ax=ax, measure=measure, scale=scale)


def animate_wavepacket_x2x0(
    wavepacket: MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv],
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    Animate the wavepacket in 3D, perpendicular to x1.

    Parameters
    ----------
    wavepacket : MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv]
    ax : Axes | None, optional
        plot axis, by default None
    measure : Literal[&quot;real&quot;, &quot;imag&quot;, &quot;abs&quot;], optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, ArtistAnimation]
    """
    return animate_wavepacket_3d(wavepacket, 1, ax=ax, measure=measure, scale=scale)


def plot_wavepacket_along_path(
    wavepacket: MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv],
    path: np.ndarray[tuple[Literal[3], int], np.dtype[np.int_]],
    *,
    ax: Axes | None = None,
    measure: Literal["real", "imag", "abs"] = "abs",
    scale: Literal["symlog", "linear"] = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the wavepacket along the given path.

    Parameters
    ----------
    wavepacket : MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv]
    path : np.ndarray[tuple[Literal[3], int], np.dtype[np.int_]]
        path to plot, as a list of x0,x1,x2 coordinate lists
    ax : Axes | None, optional
        plot axis, by default None
    measure : Literal[&quot;real&quot;, &quot;imag&quot;, &quot;abs&quot;], optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    eigenstate_momentum = unfurl_wavepacket(wavepacket)
    eigenstate = convert_eigenstate_to_position_basis(eigenstate_momentum)
    return plot_eigenstate_along_path(
        eigenstate, path, ax=ax, measure=measure, scale=scale
    )
