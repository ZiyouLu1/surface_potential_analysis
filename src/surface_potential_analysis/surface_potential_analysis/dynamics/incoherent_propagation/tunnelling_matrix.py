from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypedDict, TypeVar, overload

import numpy as np

from surface_potential_analysis.axis.axis import FundamentalAxis
from surface_potential_analysis.axis.axis_like import AxisLike
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.util.decorators import timed

from .tunnelling_basis import TunnellingSimulationBandsAxis, TunnellingSimulationBasis

if TYPE_CHECKING:
    from collections.abc import Callable

    from surface_potential_analysis._types import SingleIndexLike
    from surface_potential_analysis.operator.operator import DiagonalOperator


_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)


_B0Inv = TypeVar(
    "_B0Inv",
    bound=TunnellingSimulationBasis[Any, Any, TunnellingSimulationBandsAxis[Any]],
)


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
    shape: tuple[_L0Inv, _L1Inv], bands_axis: TunnellingSimulationBandsAxis[_L2Inv]
) -> tuple[
    FundamentalAxis[_L0Inv],
    FundamentalAxis[_L1Inv],
    TunnellingSimulationBandsAxis[_L2Inv],
]:
    return (FundamentalAxis(shape[0]), FundamentalAxis(shape[1]), bands_axis)


@timed
def get_tunnelling_a_matrix_from_function(
    shape: tuple[_L0Inv, _L1Inv],
    bands_axis: TunnellingSimulationBandsAxis[_L2Inv],
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
    tuple[
        FundamentalAxis[_L0Inv],
        FundamentalAxis[_L1Inv],
        TunnellingSimulationBandsAxis[_L2Inv],
    ]
]:
    r"""
    Given gamma as a function calculate the a matrix.

    Parameters
    ----------
    shape : _S0Inv
        shape of the simulation (nx0, nx1)
    n_bands : int
        number of bands in the simulation
    a_function : Callable[[int,int,tuple[int, int],tuple[int, int]], np.complex_]
        a_function(i, offset_i, j, offset_j), gives gamma(i,j,i,j)(\omega_{i,j})

    Returns
    -------
    TunnellingAMatrix[_S0Inv]
    """
    n_sites = np.prod(shape)
    n_bands = bands_axis.fundamental_n
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
    return {"basis": _get_a_matrix_basis(shape, bands_axis), "array": array}


_AX0Inv = TypeVar("_AX0Inv", bound=AxisLike[Any, Any])
_AX1Inv = TypeVar("_AX1Inv", bound=AxisLike[Any, Any])


def get_a_matrix_reduced_bands(
    matrix: TunnellingAMatrix[
        tuple[
            _AX0Inv,
            _AX1Inv,
            TunnellingSimulationBandsAxis[_L0Inv],
        ],
    ],
    n_bands: _L1Inv,
) -> TunnellingAMatrix[tuple[_AX0Inv, _AX1Inv, TunnellingSimulationBandsAxis[_L1Inv]]]:
    """
    Get the MMatrix with only the first n_bands included.

    Parameters
    ----------
    matrix : TunnellingMMatrix[tuple[_AX0Inv, _AX1Inv, TunnellingSimulationBandsAxis[_L0Inv]]]
    n_bands : _L1Inv

    Returns
    -------
    TunnellingMMatrix[tuple[_AX0Inv, _AX1Inv, TunnellingSimulationBandsAxis[_L1Inv]]]
    """
    util = BasisUtil(matrix["basis"])
    n_sites = np.prod(util.shape[0:2])
    return {
        "basis": (
            matrix["basis"][0],
            matrix["basis"][1],
            TunnellingSimulationBandsAxis(
                matrix["basis"][2].locations[:, 0:n_bands], matrix["basis"][2].unit_cell
            ),
        ),
        "array": matrix["array"]
        .reshape(*util.shape, *util.shape)[:, :, :n_bands, :, :, :n_bands]
        .reshape(n_bands * n_sites, n_bands * n_sites),
    }


@overload
def get_tunnelling_m_matrix(
    matrix: TunnellingAMatrix[
        tuple[
            _AX0Inv,
            _AX1Inv,
            TunnellingSimulationBandsAxis[_L0Inv],
        ]
    ],
    n_bands: _L1Inv,
) -> TunnellingMMatrix[tuple[_AX0Inv, _AX1Inv, TunnellingSimulationBandsAxis[_L1Inv]]]:
    ...


@overload
def get_tunnelling_m_matrix(
    matrix: TunnellingAMatrix[_B0Inv],
    n_bands: None = None,
) -> TunnellingMMatrix[_B0Inv]:
    ...


def get_tunnelling_m_matrix(
    matrix: TunnellingAMatrix[Any],
    n_bands: _L1Inv | None = None,
) -> TunnellingMMatrix[Any]:
    r"""
    Calculate the M matrix (M_{ij} = A_{j,i} - \delta_{i,j} \sum_k A_{i,k}).

    Parameters
    ----------
    matrix : TunnellingAMatrix

    Returns
    -------
    TunnellingMMatrix
    """
    matrix = matrix if n_bands is None else get_a_matrix_reduced_bands(matrix, n_bands)
    np.fill_diagonal(matrix["array"], 0)
    array = matrix["array"].T - np.diag(np.sum(matrix["array"], axis=1))
    return {"basis": matrix["basis"], "array": array}


def get_initial_pure_density_matrix_for_basis(
    basis: _B0Inv, idx: SingleIndexLike = 0
) -> DiagonalOperator[_B0Inv, _B0Inv]:
    """
    Given a basis get the initial pure density matrix.

    Parameters
    ----------
    basis : _B0Inv
        The basis of the density matrix
    idx : SingleIndexLike
        The index of the non-zero element, placed along the diagonal of the operator

    Returns
    -------
    DiagonalOperator[_B0Inv, _B0Inv]
    """
    util = BasisUtil(basis)
    idx = util.get_flat_index(idx) if isinstance(idx, tuple) else idx
    vector = np.zeros(util.size)
    vector[idx] = 1
    return {"basis": basis, "dual_basis": basis, "vector": vector}
