from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
from matplotlib import pyplot as plt

from surface_potential_analysis.basis.util import (
    get_fundamental_projected_k_points,
    get_fundamental_projected_x_points,
)
from surface_potential_analysis.eigenstate.conversion import (
    convert_momentum_basis_eigenstate_to_position_basis,
)
from surface_potential_analysis.eigenstate.plot import (
    animate_eigenstate_3d_x,
    plot_eigenstate_1d_x,
    plot_eigenstate_2d_k,
    plot_eigenstate_2d_x,
    plot_eigenstate_along_path,
    plot_eigenstate_difference_2d_x,
)
from surface_potential_analysis.util.util import Measure, get_measured_data
from surface_potential_analysis.wavepacket.eigenstate_conversion import (
    unfurl_wavepacket,
)

from .wavepacket import (
    MomentumBasisWavepacket3d,
    Wavepacket,
    Wavepacket3dWith2dSamples,
    get_sample_basis,
    get_wavepacket_sample_frequencies,
)

if TYPE_CHECKING:
    from matplotlib.animation import ArtistAnimation
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from surface_potential_analysis._types import SingleFlatIndexLike
    from surface_potential_analysis.axis.axis_like import AxisLike3d
    from surface_potential_analysis.basis.basis import (
        Basis,
        Basis3d,
    )
    from surface_potential_analysis.util.plot import Scale

    _NS0Inv = TypeVar("_NS0Inv", bound=int)
    _NS1Inv = TypeVar("_NS1Inv", bound=int)

    _L0Inv = TypeVar("_L0Inv", bound=int)
    _L1Inv = TypeVar("_L1Inv", bound=int)
    _L2Inv = TypeVar("_L2Inv", bound=int)

    _B3d0Inv = TypeVar("_B3d0Inv", bound=Basis3d[Any, Any, Any])

    _A3d0Inv = TypeVar("_A3d0Inv", bound=AxisLike3d[Any, Any])
    _A3d1Inv = TypeVar("_A3d1Inv", bound=AxisLike3d[Any, Any])
    _A3d2Inv = TypeVar("_A3d2Inv", bound=AxisLike3d[Any, Any])
    _S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])
    _B0Inv = TypeVar("_B0Inv", bound=Basis[Any])


def plot_wavepacket_sample_frequencies(
    wavepacket: Wavepacket3dWith2dSamples[_NS0Inv, _NS1Inv, _B3d0Inv],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the frequencies used to sample the wavepacket.

    Parameters
    ----------
    wavepacket : Wavepacket[_NS0Inv, _NS1Inv, _B3d0Inv]
    ax : Axes | None, optional
        plot axis, by default None

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    frequencies = get_wavepacket_sample_frequencies(
        wavepacket["basis"], np.array(wavepacket["vectors"].shape)[0:2]
    )[:2, :]
    (line,) = ax.plot(*frequencies.reshape(2, -1))
    line.set_marker("x")
    line.set_linestyle("")

    ax.set_xlabel("kx /$m^{-1}$")
    ax.set_ylabel("ky /$m^{-1}$")
    ax.set_title("Plot of sample points in the wavepacket")

    return fig, ax, line


def plot_wavepacket_energies_momentum(
    wavepacket: Wavepacket3dWith2dSamples[
        _NS0Inv, _NS1Inv, Basis3d[_A3d0Inv, _A3d1Inv, _A3d2Inv]
    ],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot the energy of the eigenstates in a wavepacket.

    Parameters
    ----------
    wavepacket : Wavepacket[_NS0Inv, _NS1Inv, Basis3d[_A3d0Inv, _A3d1Inv, _A3d2Inv]]
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
    """
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    basis = get_sample_basis(wavepacket["basis"], wavepacket["shape"])
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
    wavepacket: Wavepacket3dWith2dSamples[
        _NS0Inv, _NS1Inv, Basis3d[_A3d0Inv, _A3d1Inv, _A3d2Inv]
    ],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot the fourier transform of energy of the eigenstates in a wavepacket.

    Parameters
    ----------
    wavepacket : Wavepacket[_NS0Inv, _NS1Inv, Basis3d[_A3d0Inv, _A3d1Inv, _A3d2Inv]]
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

    basis = get_sample_basis(wavepacket["basis"], wavepacket["shape"])
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


def plot_wavepacket_1d_x(
    wavepacket: Wavepacket[_S0Inv, _B0Inv],
    idx: tuple[int, ...] | None = None,
    axis: Literal[0, 1, 2, -1, -2, -3] = 0,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot a wavepacket in 2d along the given axis.

    Parameters
    ----------
    wavepacket : MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv]
    idx : tuple[int, int], optional
        index through axis perpendicular to axis, by default (0,0)
    axis : Literal[0, 1, 2,, optional
        axis along which to plot, by default 2
    ax : Axes | None, optional
        plot axis, by default None
    measure : Measure, optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    eigenstate = unfurl_wavepacket(wavepacket)
    return plot_eigenstate_1d_x(
        eigenstate, idx, axis, ax=ax, measure=measure, scale=scale
    )


def plot_wavepacket_x0(
    wavepacket: MomentumBasisWavepacket3d[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv],
    idx: tuple[int, int] = (0, 0),
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot a wavepacket in 2d along the x0 axis.

    Parameters
    ----------
    wavepacket : MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv]
    idx : tuple[int, int], optional
        index through x1, x2 axis, by default (0,0)
    ax : Axes | None, optional
        plot axis, by default None
    measure : Measure, optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    eigenstate = unfurl_wavepacket(wavepacket)
    return plot_eigenstate_1d_x(eigenstate, idx, 0, ax=ax, measure=measure, scale=scale)


def plot_wavepacket_x1(
    wavepacket: MomentumBasisWavepacket3d[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv],
    idx: tuple[int, int] = (0, 0),
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot a wavepacket in 2d along the x1 axis.

    Parameters
    ----------
    wavepacket : MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv]
    idx : tuple[int, int], optional
        index through x2, x0 axis, by default (0,0)
    ax : Axes | None, optional
        plot axis, by default None
    measure : Measure, optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    eigenstate = unfurl_wavepacket(wavepacket)
    return plot_eigenstate_1d_x(eigenstate, idx, 1, ax=ax, measure=measure, scale=scale)


def plot_wavepacket_x2(
    wavepacket: MomentumBasisWavepacket3d[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv],
    idx: tuple[int, int] = (0, 0),
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot a wavepacket in 2d along the x2 axis.

    Parameters
    ----------
    wavepacket : MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv]
    idx : tuple[int, int], optional
        index through x0, x1 axis, by default (0,0)
    ax : Axes | None, optional
        plot axis, by default None
    measure : Measure, optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    eigenstate = unfurl_wavepacket(wavepacket)
    return plot_eigenstate_1d_x(eigenstate, idx, 2, ax=ax, measure=measure, scale=scale)


def plot_wavepacket_2d_k(
    wavepacket: MomentumBasisWavepacket3d[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv],
    idx: SingleFlatIndexLike,
    kz_axis: Literal[0, 1, 2, -1, -2, -3],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot wavepacket in 2D at idx along the given axis in momentum.

    Parameters
    ----------
    wavepacket : MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv]
        Wavepacket in momentum basis
    idx : SingleFlatIndexLike
        index along z_axis
    kz_axis : Literal[0, 1, 2]
        kz_axis, perpendicular to plotted direction
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
    eigenstate = unfurl_wavepacket(wavepacket)
    return plot_eigenstate_2d_k(
        eigenstate, idx, kz_axis, ax=ax, measure=measure, scale=scale
    )


def plot_wavepacket_k0k1(
    wavepacket: MomentumBasisWavepacket3d[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv],
    k2_idx: SingleFlatIndexLike,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot wavepacket perpendicular to the k2 axis.

    Parameters
    ----------
    wavepacket : MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv]
        Wavepacket in momentum basis
    k2_idx : SingleFlatIndexLike
        index along k2 axis
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
    return plot_wavepacket_2d_k(
        wavepacket, k2_idx, 2, ax=ax, measure=measure, scale=scale
    )


def plot_wavepacket_k1k2(
    wavepacket: MomentumBasisWavepacket3d[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv],
    k0_idx: SingleFlatIndexLike,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot wavepacket perpendicular to the k0 axis.

    Parameters
    ----------
    wavepacket : MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv]
        Wavepacket in momentum basis
    k0_idx : SingleFlatIndexLike
        index along k0 axis
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
    return plot_wavepacket_2d_k(
        wavepacket, k0_idx, 0, ax=ax, measure=measure, scale=scale
    )


def plot_wavepacket_k2k0(
    wavepacket: MomentumBasisWavepacket3d[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv],
    k1_idx: SingleFlatIndexLike,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot wavepacket perpendicular to the k1 axis.

    Parameters
    ----------
    wavepacket : MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv]
        Wavepacket in momentum basis
    k1_idx : SingleFlatIndexLike
        index along k1 axis
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
    return plot_wavepacket_2d_k(
        wavepacket, k1_idx, 1, ax=ax, measure=measure, scale=scale
    )


def plot_wavepacket_2d_x(
    wavepacket: MomentumBasisWavepacket3d[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv],
    idx: SingleFlatIndexLike,
    z_axis: Literal[0, 1, 2, -1, -2, -3],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot wavepacket in 2D at idx along the given axis.

    Parameters
    ----------
    wavepacket : MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv]
        Wavepacket in momentum basis
    idx : SingleFlatIndexLike
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
    eigenstate = unfurl_wavepacket(wavepacket)
    return plot_eigenstate_2d_x(
        eigenstate, idx, z_axis, ax=ax, measure=measure, scale=scale
    )


def plot_wavepacket_x0x1(
    wavepacket: MomentumBasisWavepacket3d[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv],
    x2_idx: SingleFlatIndexLike,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot wavepacket perpendicular to the x2 axis.

    Parameters
    ----------
    wavepacket : MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv]
        Wavepacket in momentum basis
    x2_idx : SingleFlatIndexLike
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
    return plot_wavepacket_2d_x(
        wavepacket, x2_idx, 2, ax=ax, measure=measure, scale=scale
    )


def plot_wavepacket_x1x2(
    wavepacket: MomentumBasisWavepacket3d[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv],
    x0_idx: SingleFlatIndexLike,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot wavepacket perpendicular to the x0 axis.

    Parameters
    ----------
    wavepacket : MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv]
        Wavepacket in momentum basis
    x0_idx : SingleFlatIndexLike
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
    return plot_wavepacket_2d_x(
        wavepacket, x0_idx, 0, ax=ax, measure=measure, scale=scale
    )


def plot_wavepacket_x2x0(
    wavepacket: MomentumBasisWavepacket3d[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv],
    x1_idx: SingleFlatIndexLike,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot wavepacket perpendicular to the x1 axis.

    Parameters
    ----------
    wavepacket : MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv]
        Wavepacket in momentum basis
    x1_idx : SingleFlatIndexLike
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
    return plot_wavepacket_2d_x(
        wavepacket, x1_idx, 1, ax=ax, measure=measure, scale=scale
    )


def plot_wavepacket_difference_2d_x(
    wavepacket_0: MomentumBasisWavepacket3d[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv],
    wavepacket_1: MomentumBasisWavepacket3d[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv],
    idx: SingleFlatIndexLike,
    z_axis: Literal[0, 1, 2, -1, -2, -3],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot the difference between two wavepackets in 2D.

    Parameters
    ----------
    wavepacket_0 : MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv]
    wavepacket_1 : MomentumBasisWavepacket[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv]
    idx : SingleFlatIndexLike
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
    eigenstate_momentum_0 = unfurl_wavepacket(wavepacket_0)
    eigenstate_0 = convert_momentum_basis_eigenstate_to_position_basis(
        eigenstate_momentum_0
    )

    eigenstate_momentum_1 = unfurl_wavepacket(wavepacket_1)
    eigenstate_1 = convert_momentum_basis_eigenstate_to_position_basis(
        eigenstate_momentum_1
    )

    return plot_eigenstate_difference_2d_x(
        eigenstate_0, eigenstate_1, idx, z_axis, ax=ax, measure=measure, scale=scale
    )


def animate_wavepacket_3d_x(
    wavepacket: MomentumBasisWavepacket3d[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv],
    z_axis: Literal[0, 1, 2, -1, -2, -3],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
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
    eigenstate = convert_momentum_basis_eigenstate_to_position_basis(
        eigenstate_momentum
    )
    return animate_eigenstate_3d_x(
        eigenstate, z_axis, ax=ax, measure=measure, scale=scale
    )


def animate_wavepacket_x0x1(
    wavepacket: MomentumBasisWavepacket3d[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
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
    return animate_wavepacket_3d_x(wavepacket, 2, ax=ax, measure=measure, scale=scale)


def animate_wavepacket_x1x2(
    wavepacket: MomentumBasisWavepacket3d[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
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
    return animate_wavepacket_3d_x(wavepacket, 0, ax=ax, measure=measure, scale=scale)


def animate_wavepacket_x2x0(
    wavepacket: MomentumBasisWavepacket3d[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
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
    return animate_wavepacket_3d_x(wavepacket, 1, ax=ax, measure=measure, scale=scale)


def plot_wavepacket_along_path(
    wavepacket: MomentumBasisWavepacket3d[_NS0Inv, _NS1Inv, _L0Inv, _L1Inv, _L2Inv],
    path: np.ndarray[tuple[Literal[3], int], np.dtype[np.int_]],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
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
    eigenstate = convert_momentum_basis_eigenstate_to_position_basis(
        eigenstate_momentum
    )
    return plot_eigenstate_along_path(
        eigenstate, path, ax=ax, measure=measure, scale=scale
    )
