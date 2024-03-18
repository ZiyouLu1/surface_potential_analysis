from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from surface_potential_analysis.basis.basis import FundamentalBasis
from surface_potential_analysis.basis.stacked_basis import StackedBasis
from surface_potential_analysis.kernel.kernel import (
    get_diagonal_noise_kernel,
    get_noise_kernel,
    get_single_factorized_noise_operators,
    get_single_factorized_noise_operators_diagonal,
)
from surface_potential_analysis.util.plot import (
    Scale,
    _get_lim,
    _get_norm_with_lim,
    _get_scale_with_lim,
    get_figure,
)
from surface_potential_analysis.util.util import Measure, get_measured_data

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.figure import Figure

    from surface_potential_analysis.kernel.kernel import (
        DiagonalNoiseKernel,
        NoiseKernel,
    )


def plot_diagonal_kernel(
    kernel: DiagonalNoiseKernel[Any, Any, Any, Any],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot the elements of a diagonal kernel.

    Parameters
    ----------
    kernel : DiagonalNoiseKernel[Any, Any, Any, Any]
    ax : Axes | None, optional
        axis, by default None
    measure : Measure, optional
        measure, by default "abs"
    scale : Scale, optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]

    """
    fig, ax = get_figure(ax)
    n_states = kernel["basis"][0].shape[0]
    data = kernel["data"].reshape(n_states, n_states)
    measured_data = get_measured_data(data, measure)

    mesh = ax.pcolormesh(measured_data)
    clim = _get_lim((None, None), measure, measured_data)
    norm = _get_norm_with_lim(scale, clim)
    mesh.set_norm(norm)
    mesh.set_clim(*clim)
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(mesh, ax=ax, format="%4.1e")
    return fig, ax, mesh


def plot_kernel(
    kernel: NoiseKernel[Any, Any, Any, Any],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    fig, ax = get_figure(ax)
    data = kernel["data"].reshape(kernel["basis"].shape)
    measured_data = get_measured_data(data, measure)

    mesh = ax.pcolormesh(measured_data)
    clim = _get_lim((None, None), measure, measured_data)
    norm = _get_norm_with_lim(scale, clim)
    mesh.set_norm(norm)
    mesh.set_clim(*clim)
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(mesh, ax=ax, format="%4.1e")
    return fig, ax, mesh


def plot_kernel_sparsity(
    kernel: NoiseKernel[Any, Any, Any, Any],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes]:
    fig, ax = get_figure(ax)

    operators = get_single_factorized_noise_operators(kernel)

    data = get_measured_data(operators["eigenvalue"], measure)
    bins = np.logspace(
        np.log10(np.min(data)), np.log10(np.max(data)), 10000, dtype=np.float64
    )
    values, bins = np.histogram(data, bins=bins)

    cumulative = np.cumsum(values)
    ax.plot(bins[:-1], cumulative)

    ax.set_yscale(scale)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_xscale("log")

    return fig, ax


def plot_kernel_truncation_error(
    kernel: NoiseKernel[Any, Any, Any, Any],
    *,
    truncations: list[int] | None = None,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes]:
    fig, ax = get_figure(ax)

    operators = get_single_factorized_noise_operators(kernel)
    arg_sort = np.argsort(np.abs(operators["eigenvalue"]))

    n_operators = operators["basis"][0].n
    data = list[np.float64]()
    truncations = (
        list(range(0, n_operators, n_operators // 100))
        if truncations is None
        else truncations
    )

    for truncation in truncations:
        args = arg_sort[truncation::]
        truncated = get_noise_kernel(
            {
                "basis": StackedBasis(
                    FundamentalBasis(n_operators - truncation), operators["basis"][1]
                ),
                "data": operators["data"].reshape(n_operators, -1)[args].ravel(),
                "eigenvalue": operators["eigenvalue"][args],
            }
        )
        data.append(np.linalg.norm(truncated["data"] - kernel["data"]))

    data.append(np.linalg.norm(kernel["data"]))
    truncations.append(n_operators)

    ax.plot(truncations, data)
    ax.set_yscale(scale)

    return fig, ax


def plot_diagonal_kernel_truncation_error(
    kernel: DiagonalNoiseKernel[Any, Any, Any, Any],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes]:
    fig, ax = get_figure(ax)

    operators = get_single_factorized_noise_operators_diagonal(kernel)
    arg_sort = np.argsort(np.abs(operators["eigenvalue"]))

    n_operators = operators["basis"][0].n
    data = []
    for truncation in range(n_operators):
        args = arg_sort[truncation::]
        truncated = get_diagonal_noise_kernel(
            {
                "basis": StackedBasis(
                    FundamentalBasis(n_operators - truncation), operators["basis"][1]
                ),
                "data": operators["data"].reshape(n_operators, -1)[args].ravel(),
                "eigenvalue": operators["eigenvalue"][args],
            }
        )
        data.append(np.linalg.norm(truncated["data"] - kernel["data"]))
    data.append(np.linalg.norm(kernel["data"]))

    ax.plot(data)
    ax.set_yscale(_get_scale_with_lim(scale, ax.get_ylim()))

    return fig, ax