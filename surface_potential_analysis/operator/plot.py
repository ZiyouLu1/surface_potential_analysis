from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from surface_potential_analysis.basis.basis_like import BasisLike
from surface_potential_analysis.state_vector.eigenstate_calculation import (
    calculate_eigenvectors,
    calculate_eigenvectors_hermitian,
)
from surface_potential_analysis.state_vector.eigenvalue_list_plot import (
    plot_eigenstate_occupations,
    plot_eigenvalues,
)
from surface_potential_analysis.util.plot import (
    Scale,
    get_figure,
)
from surface_potential_analysis.util.util import (
    Measure,
    get_measured_data,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from surface_potential_analysis.basis.basis_like import BasisLike
    from surface_potential_analysis.operator.operator import SingleBasisOperator

    from .operator import Operator


def plot_operator_sparsity(
    operator: Operator[BasisLike[Any, Any], BasisLike[Any, Any]],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes]:
    fig, ax = get_figure(ax)
    measured = get_measured_data(operator["data"], measure)

    values, bins = np.histogram(
        measured,
        bins=np.logspace(-22, np.log10(np.max(measured)), 10000),
    )

    cumulative = np.cumsum(values)
    ax.plot(bins[:-1], cumulative)

    ax.set_yscale(scale)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_xscale("log")

    return fig, ax


def plot_eigenstate_occupations_operator(
    hamiltonian: SingleBasisOperator[BasisLike[Any, Any]],
    temperature: float,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the expected occupation of eigenstates at the given temperature.

    Parameters
    ----------
    eigenstates : EigenstateList[BasisLike[Any, Any], BasisLike[Any, Any]]
    temperature : float
    ax : Axes | None, optional
        ax, by default None
    scale : Scale, optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    eigenstates = calculate_eigenvectors(hamiltonian)
    return plot_eigenstate_occupations(eigenstates, temperature, ax=ax, scale=scale)


def plot_eigenvalues_operator(
    hamiltonian: SingleBasisOperator[BasisLike[Any, Any]],
    *,
    hermitian: bool = False,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "abs",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the expected occupation of eigenstates at the given temperature.

    Parameters
    ----------
    eigenstates : EigenstateList[BasisLike[Any, Any], BasisLike[Any, Any]]
    temperature : float
    ax : Axes | None, optional
        ax, by default None
    scale : Scale, optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    eigenstates = (
        calculate_eigenvectors_hermitian(hamiltonian)
        if hermitian
        else calculate_eigenvectors(hamiltonian)
    )
    return plot_eigenvalues(eigenstates, ax=ax, scale=scale, measure=measure)
