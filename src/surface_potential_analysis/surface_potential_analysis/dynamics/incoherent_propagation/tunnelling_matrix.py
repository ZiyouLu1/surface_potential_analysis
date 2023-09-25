from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Literal, TypedDict, TypeVar, overload

import numpy as np

from surface_potential_analysis.basis.basis import FundamentalBasis
from surface_potential_analysis.basis.basis_like import BasisLike
from surface_potential_analysis.basis.stacked_basis import StackedBasis
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.dynamics.tunnelling_basis import (
    TunnellingSimulationBandsBasis,
    TunnellingSimulationBasis,
    get_basis_from_shape,
)
from surface_potential_analysis.dynamics.util import build_hop_operator, get_hop_shift
from surface_potential_analysis.util.decorators import timed

if TYPE_CHECKING:
    from collections.abc import Callable

    from surface_potential_analysis.basis.stacked_basis import StackedBasisLike
    from surface_potential_analysis.operator.operator import DiagonalOperator
    from surface_potential_analysis.operator.operator_list import DiagonalOperatorList
    from surface_potential_analysis.probability_vector.probability_vector import (
        ProbabilityVector,
        ProbabilityVectorList,
    )
    from surface_potential_analysis.types import SingleIndexLike

    _L1Inv = TypeVar("_L1Inv", bound=int)
    _L2Inv = TypeVar("_L2Inv", bound=int)
    _L3Inv = TypeVar("_L3Inv", bound=int)

    _AX0Inv = TypeVar("_AX0Inv", bound=BasisLike[Any, Any])
    _AX1Inv = TypeVar("_AX1Inv", bound=BasisLike[Any, Any])


_L0Inv = TypeVar("_L0Inv", bound=int)
_AX2Inv = TunnellingSimulationBandsBasis[Any]

_B0Inv = TypeVar("_B0Inv", bound=BasisLike[Any, Any])
_B1Inv = TypeVar(
    "_B1Inv",
    bound=TunnellingSimulationBasis[Any, Any, TunnellingSimulationBandsBasis[Any]],
)


class TunnellingJumpMatrix(TypedDict, Generic[_L0Inv]):
    """Gives the jump between site n0 to n1 with a hop idx."""

    axis: TunnellingSimulationBandsBasis[_L0Inv]
    array: np.ndarray[tuple[int, int, Literal[9]], np.dtype[np.float_]]


# Note A has the reverse convention
class TunnellingAMatrix(TypedDict, Generic[_B1Inv]):
    """
    A_{i,j}.

    Indexed such that A.reshape(*shape, n_bands, *shape, n_bands)[i0,j0,n0,i1,j1,n1]
    gives a from site i=i0,j0,n0 to site j=i1,j1,n1.
    """

    basis: _B1Inv
    array: np.ndarray[tuple[int, int], np.dtype[np.float_]]


class TunnellingMMatrix(TypedDict, Generic[_B1Inv]):
    """
    M_{i,j}.

    Indexed such that A.reshape(*shape, n_bands, *shape, n_bands)[i0,j0,n0,i1,j1,n1]
    gives a from site i=i0,j0,n0 to site j=i1,j1,n1.
    """

    basis: _B1Inv
    array: np.ndarray[tuple[int, int], np.dtype[np.float_]]


FundamentalTunnellingAMatrixBasis = TunnellingSimulationBasis[
    FundamentalBasis[Literal[3]],
    FundamentalBasis[Literal[3]],
    _AX2Inv,
]


def get_jump_matrix_from_a_matrix(
    matrix: TunnellingAMatrix[
        TunnellingSimulationBasis[
            _AX0Inv, _AX1Inv, TunnellingSimulationBandsBasis[_L2Inv]
        ]
    ]
) -> TunnellingJumpMatrix[_L2Inv]:
    """
    Given an A Matrix, get a jump matrix.

    Parameters
    ----------
    matrix : TunnellingAMatrix[ TunnellingSimulationBasis[ _AX0Inv, _AX1Inv, TunnellingSimulationBandsBasis[_L2Inv] ] ]

    Returns
    -------
    TunnellingJumpMatrix[_L2Inv]
    """
    n_bands = matrix["basis"][2].fundamental_n
    util = BasisUtil(matrix["basis"])
    stacked = matrix["array"].reshape(*util.shape, *util.shape)
    array = np.zeros((n_bands, n_bands, 9))
    for n_0 in range(n_bands):
        for n_1 in range(n_bands):
            for hop in range(9):
                hop_shift = get_hop_shift(hop, 2)
                # A matrix uses the reverse order
                hop_val = stacked[0, 0, n_0, hop_shift[0], hop_shift[1], n_1]
                array[n_0, n_1, hop] = hop_val

    return {"axis": matrix["basis"][2], "array": array}


@overload
def get_a_matrix_from_jump_matrix(
    matrix: TunnellingJumpMatrix[_L2Inv],
    shape: tuple[_L0Inv, _L1Inv],
    *,
    n_bands: None = None,
) -> TunnellingAMatrix[
    StackedBasisLike[
        FundamentalBasis[_L0Inv],
        FundamentalBasis[_L1Inv],
        TunnellingSimulationBandsBasis[_L2Inv],
    ]
]:
    ...


@overload
def get_a_matrix_from_jump_matrix(
    matrix: TunnellingJumpMatrix[_L2Inv],
    shape: tuple[_L0Inv, _L1Inv],
    *,
    n_bands: _L3Inv,
) -> TunnellingAMatrix[
    StackedBasisLike[
        FundamentalBasis[_L0Inv],
        FundamentalBasis[_L1Inv],
        TunnellingSimulationBandsBasis[_L3Inv],
    ]
]:
    ...


def get_a_matrix_from_jump_matrix(
    matrix: TunnellingJumpMatrix[_L2Inv],
    shape: tuple[_L0Inv, _L1Inv],
    *,
    n_bands: Any = None,
) -> TunnellingAMatrix[
    StackedBasisLike[
        FundamentalBasis[_L0Inv],
        FundamentalBasis[_L1Inv],
        TunnellingSimulationBandsBasis[Any],
    ]
]:
    """
    Given a jump matrix get an a matrix.

    Parameters
    ----------
    matrix : TunnellingJumpMatrix[_L2Inv]
    shape : tuple[_L0Inv, _L1Inv]
    n_bands : int, optional
        number of bands, by default None

    Returns
    -------
    TunnellingAMatrix[ tuple[ FundamentalBasis[_L0Inv], FundamentalBasis[_L1Inv], TunnellingSimulationBandsBasis[int], ] ]
    """
    n_bands = matrix["axis"].fundamental_n if n_bands is None else n_bands
    final_basis = get_basis_from_shape(shape, n_bands, matrix["axis"])
    final_util = BasisUtil(final_basis)

    (n_x0, n_x1) = shape
    array = np.zeros((*final_util.shape, *final_util.shape))
    for n_0 in range(n_bands):
        for n_1 in range(n_bands):
            for hop in range(9):
                hop_val = matrix["array"][n_0, n_1, hop]
                operator = hop_val * build_hop_operator(hop, (n_x0, n_x1))
                array[:, :, n_1, :, :, n_0] += operator
    # A matrix uses the reverse convention for array, ie n_0 first
    return {
        "basis": final_basis,
        "array": array.reshape(final_util.n, final_util.n).T,
    }


def resample_tunnelling_a_matrix(
    matrix: TunnellingAMatrix[TunnellingSimulationBasis[_AX0Inv, _AX1Inv, _AX2Inv]],
    shape: tuple[_L0Inv, _L1Inv],
    n_bands: _L2Inv,
) -> TunnellingAMatrix[
    StackedBasisLike[
        FundamentalBasis[_L0Inv],
        FundamentalBasis[_L1Inv],
        TunnellingSimulationBandsBasis[_L2Inv],
    ]
]:
    """
    Given a tunnelling a matrix in at least a 3x3 grid, generate the full matrix.

    Uses the symmetry properties of the a matrix

    Parameters
    ----------
    matrix : TunnellingAMatrix[FundamentalTunnellingAMatrixBasis[_AX2Inv]]
    shape : tuple[_L0Inv, _L1Inv]

    Returns
    -------
    TunnellingAMatrix[tuple[FundamentalBasis[_L0Inv], FundamentalBasis[_L1Inv], _AX2Inv]]
    """
    jump_matrix = get_jump_matrix_from_a_matrix(matrix)
    return get_a_matrix_from_jump_matrix(jump_matrix, shape, n_bands=n_bands)


@timed
def get_jump_matrix_from_function(
    bands_axis: TunnellingSimulationBandsBasis[_L2Inv],
    jump_function: Callable[[int, int, int], float],
) -> TunnellingJumpMatrix[_L2Inv]:
    r"""
    Given gamma as a function calculate the a matrix.

    Parameters
    ----------
    shape : _S0Inv
        shape of the simulation (nx0, nx1)
    n_bands : int
        number of bands in the simulation
    a_function : Callable[[int, int, int], float]
        a_function(i, j, hop_idx)

    Returns
    -------
    TunnellingAMatrix[_S0Inv]
    """
    n_bands = bands_axis.fundamental_n
    array = np.zeros((n_bands, n_bands, 9))
    for n0 in range(n_bands):
        for n1 in range(n_bands):
            for d1 in range(9):
                array[n0, n1, d1] = jump_function((n0), n1, d1)
    return {"axis": bands_axis, "array": array}


def get_a_matrix_reduced_bands(
    matrix: TunnellingAMatrix[
        StackedBasisLike[
            _AX0Inv,
            _AX1Inv,
            TunnellingSimulationBandsBasis[_L0Inv],
        ],
    ],
    n_bands: _L1Inv,
) -> TunnellingAMatrix[
    StackedBasisLike[_AX0Inv, _AX1Inv, TunnellingSimulationBandsBasis[_L1Inv]]
]:
    """
    Get the MMatrix with only the first n_bands included.

    Parameters
    ----------
    matrix : TunnellingMMatrix[tuple[_AX0Inv, _AX1Inv, TunnellingSimulationBandsBasis[_L0Inv]]]
    n_bands : _L1Inv

    Returns
    -------
    TunnellingMMatrix[tuple[_AX0Inv, _AX1Inv, TunnellingSimulationBandsBasis[_L1Inv]]]
    """
    util = BasisUtil(matrix["basis"])
    n_sites = np.prod(util.shape[0:2])
    return {
        "basis": StackedBasis(
            matrix["basis"][0],
            matrix["basis"][1],
            TunnellingSimulationBandsBasis(
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
        StackedBasisLike[
            _AX0Inv,
            _AX1Inv,
            TunnellingSimulationBandsBasis[_L0Inv],
        ]
    ],
    n_bands: _L1Inv,
) -> TunnellingMMatrix[
    StackedBasisLike[_AX0Inv, _AX1Inv, TunnellingSimulationBandsBasis[_L1Inv]]
]:
    ...


@overload
def get_tunnelling_m_matrix(
    matrix: TunnellingAMatrix[_B1Inv],
    n_bands: None = None,
) -> TunnellingMMatrix[_B1Inv]:
    ...


def get_tunnelling_m_matrix(
    matrix: TunnellingAMatrix[Any],
    n_bands: int | None = None,
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
    basis: _B1Inv, idx: SingleIndexLike = 0
) -> DiagonalOperator[_B1Inv, _B1Inv]:
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
    vector = np.zeros(basis.n, dtype=np.complex_)
    vector[idx] = 1
    return {"basis": StackedBasis(basis, basis), "data": vector}


def density_matrix_as_probability(
    matrix: DiagonalOperator[_B1Inv, _B1Inv]
) -> ProbabilityVector[_B1Inv]:
    """
    Get the probability of each state in eh density matrix.

    Parameters
    ----------
    matrix : DiagonalOperator[_B0Inv, _B0Inv]

    Returns
    -------
    ProbabilityVector[_B0Inv]
    """
    return {"basis": matrix["basis"], "data": np.real(matrix["data"])}  # type: ignore[typeddict-item]


def density_matrix_list_as_probabilities(
    matrix: DiagonalOperatorList[_B0Inv, _B1Inv, _B1Inv]
) -> ProbabilityVectorList[_B0Inv, _B1Inv]:
    """
    Get the probability of each state in the density matrix.

    Parameters
    ----------
    matrix : DiagonalOperatorList[_B0Inv, _B0Inv, _L0Inv]

    Returns
    -------
    ProbabilityVectorList[_B0Inv, _L0Inv]
    """
    return {
        "basis": StackedBasis(matrix["basis"][0], matrix["basis"][1][0]),
        "data": np.real(matrix["data"]),
    }
