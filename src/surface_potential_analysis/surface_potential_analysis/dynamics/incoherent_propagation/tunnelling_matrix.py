from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypedDict, TypeVar

import numpy as np

from surface_potential_analysis.axis.axis import FundamentalAxis
from surface_potential_analysis.axis.axis_like import AxisLike
from surface_potential_analysis.util.decorators import timed

if TYPE_CHECKING:
    from collections.abc import Callable


_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)

TunnellingSimulationBasis = tuple[
    AxisLike[Any, Any], AxisLike[Any, Any], AxisLike[Any, Any]
]
"""
Basis used to represent the tunnelling simulation state

First two axes represent the x,y location of the unit cell, last axis is the band
"""


_B0Inv = TypeVar("_B0Inv", bound=TunnellingSimulationBasis)


class TunnellingAMatrix(TypedDict, Generic[_B0Inv]):
    """
    A_{i,j}.

    Indexed such that A.reshape(*shape, n_bands, *shape, n_bands)[i0,j0,n0,i1,j1,n1]
    gives a from site i=i0,j0,n0 to site j=i1,j1,n1.
    """

    basis: _B0Inv
    array: np.ndarray[tuple[int, int], np.dtype[np.float_]]


class TunnellingMMatrix(TypedDict, Generic[_B0Inv]):
    """
    M_{i,j}.

    Indexed such that A.reshape(*shape, n_bands, *shape, n_bands)[i0,j0,n0,i1,j1,n1]
    gives a from site i=i0,j0,n0 to site j=i1,j1,n1.
    """

    basis: _B0Inv
    array: np.ndarray[tuple[int, int], np.dtype[np.float_]]


def _get_a_matrix_basis(
    shape: tuple[_L0Inv, _L1Inv], n_bands: _L2Inv
) -> tuple[FundamentalAxis[_L0Inv], FundamentalAxis[_L1Inv], FundamentalAxis[_L2Inv]]:
    return (
        FundamentalAxis(shape[0]),
        FundamentalAxis(shape[1]),
        FundamentalAxis(n_bands),
    )


@timed
def get_tunnelling_a_matrix_from_function(
    shape: tuple[_L0Inv, _L1Inv],
    n_bands: _L2Inv,
    a_function: Callable[
        [
            int,
            int,
            tuple[int, int],
            tuple[int, int],
        ],
        float,
    ],
) -> TunnellingAMatrix[
    tuple[FundamentalAxis[_L0Inv], FundamentalAxis[_L1Inv], FundamentalAxis[_L2Inv]]
]:
    r"""
    Given gamma as a function calculate the a matrix.

    Parameters
    ----------
    shape : _S0Inv
        shape of the simulation (nx0, nx1)
    n_bands : int
        number of bands in the simulation
    a_function : Callable[ [ SupportsInt, tuple[SupportsInt, SupportsInt], SupportsInt, tuple[SupportsInt, SupportsInt], ], np.complex_, ]
        a_function(i, offset_i, j, offset_j), gives gamma(i,j,i,j)(\omega_{i,j})

    Returns
    -------
    TunnellingAMatrix[_S0Inv]
    """
    n_sites = np.prod(shape)
    array = np.zeros((n_sites * n_bands, n_sites * n_bands))
    for i in range(array.shape[0]):
        for n1 in range(n_bands):
            for d1 in range(9):
                (i0, j0, n0) = np.unravel_index(i, (*shape, n_bands))
                d1_stacked = np.unravel_index(d1, (3, 3)) - np.array([1, 1])
                (i1, j1) = (i0 + d1_stacked[0], j0 + d1_stacked[1])
                j = np.ravel_multi_index((i1, j1, n1), (*shape, n_bands), mode="wrap")

                array[i, j] = a_function(
                    int(n0), n1, (0, 0), (d1_stacked[0], d1_stacked[1])
                )
    return {"basis": _get_a_matrix_basis(shape, n_bands), "array": array}


@timed
def get_tunnelling_m_matrix(
    matrix: TunnellingAMatrix[_B0Inv],
) -> TunnellingMMatrix[_B0Inv]:
    r"""
    Calculate the M matrix (M_{ij} = A_{j,i} - \delta_{i,j} \sum_k A_{i,k}).

    Parameters
    ----------
    matrix : TunnellingAMatrix

    Returns
    -------
    TunnellingMMatrix
    """
    np.fill_diagonal(matrix["array"], 0)
    array = matrix["array"].T - np.diag(np.sum(matrix["array"], axis=1))
    return {"basis": matrix["basis"], "array": array}
