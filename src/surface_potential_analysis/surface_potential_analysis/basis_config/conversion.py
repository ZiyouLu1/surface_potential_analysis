from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np

from surface_potential_analysis.basis.conversion import (
    basis_as_fundamental_momentum_basis,
    basis_as_fundamental_position_basis,
    basis_as_single_point_basis,
    get_basis_conversion_matrix,
    get_rotated_basis,
)

from .util import BasisConfigUtil

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import (
        FundamentalMomentumBasis,
        FundamentalPositionBasis,
    )
    from surface_potential_analysis.basis.basis_like import BasisLike, BasisVector

    from .basis_config import BasisConfig

    _BX0Inv = TypeVar("_BX0Inv", bound=BasisLike[Any, Any])
    _BX1Inv = TypeVar("_BX1Inv", bound=BasisLike[Any, Any])
    _BX2Inv = TypeVar("_BX2Inv", bound=BasisLike[Any, Any])

    _BC0Inv = TypeVar("_BC0Inv", bound=BasisConfig[Any, Any, Any])
    _BC1Inv = TypeVar("_BC1Inv", bound=BasisConfig[Any, Any, Any])
    _S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])


def convert_vector(
    vector: np.ndarray[_S0Inv, np.dtype[np.complex_]],
    initial_basis: _BC0Inv,
    final_basis: _BC1Inv,
    axis: int = -1,
) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
    """
    Convert a vector, expressed in terms of the given basis from_config in the basis to_config.

    Parameters
    ----------
    vector : np.ndarray[tuple[int], np.dtype[np.complex_]]
        the vector to convert
    from_config : _BC0Inv
    to_config : _BC1Inv
    axis : int, optional
        axis along which to convert, by default -1

    Returns
    -------
    np.ndarray[tuple[int], np.dtype[np.complex_]]
    """
    util = BasisConfigUtil(initial_basis)
    swapped = vector.swapaxes(axis, -1)
    stacked = swapped.reshape(*swapped.shape[:-1], *util.shape)
    last_axis = swapped.ndim - 1
    for i in range(3):
        matrix = get_basis_conversion_matrix(initial_basis[i], final_basis[i])
        # Each product gets moved to the end,
        # so "last_axis" of stacked always corresponds to the ith axis
        stacked = np.tensordot(stacked, matrix, axes=([last_axis], [0]))

    return stacked.reshape(*swapped.shape[:-1], -1).swapaxes(axis, -1)


def convert_matrix(
    matrix: np.ndarray[tuple[int, int], np.dtype[np.complex_]],
    initial_basis: _BC0Inv,
    final_basis: _BC1Inv,
) -> np.ndarray[tuple[int, int], np.dtype[np.complex_]]:
    """
    Convert a matrix from initial_basis to final_basis.

    Parameters
    ----------
    matrix : np.ndarray[tuple[int, int], np.dtype[np.complex_]]
    initial_basis : _BC0Inv
    final_basis : _BC1Inv

    Returns
    -------
    np.ndarray[tuple[int, int], np.dtype[np.complex_]]
    """
    util = BasisConfigUtil(initial_basis)
    stacked = matrix.reshape(*util.shape, *util.shape)
    for i in range(3):
        matrix = get_basis_conversion_matrix(initial_basis[i], final_basis[i])
        # Each product gets moved to the end,
        # so the 0th index of stacked corresponds to the ith axis
        stacked = np.tensordot(stacked, matrix, axes=([0], [0]))

    for i in range(3):
        matrix = get_basis_conversion_matrix(initial_basis[i], final_basis[i])
        matrix_conj = np.conj(matrix)
        # Each product gets moved to the end,
        # so the 0th index of stacked corresponds to the ith axis
        stacked = np.tensordot(stacked, matrix_conj, axes=([0], [0]))
    final_size = BasisConfigUtil(final_basis).size
    return stacked.reshape(final_size, final_size)


_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)

_LF0Inv = TypeVar("_LF0Inv", bound=int)
_LF1Inv = TypeVar("_LF1Inv", bound=int)
_LF2Inv = TypeVar("_LF2Inv", bound=int)


def basis_config_as_fundamental_momentum_basis_config(
    config: BasisConfig[
        BasisLike[_LF0Inv, _L0Inv],
        BasisLike[_LF1Inv, _L1Inv],
        BasisLike[_LF2Inv, _L2Inv],
    ]
) -> BasisConfig[
    FundamentalMomentumBasis[_LF0Inv],
    FundamentalMomentumBasis[_LF1Inv],
    FundamentalMomentumBasis[_LF2Inv],
]:
    """
    Get the fundamental momentum basis for a given basis.

    Parameters
    ----------
    self : BasisConfigUtil[tuple[BasisLike[_LF0Inv, _L0Inv], BasisLike[_LF1Inv, _L1Inv], BasisLike[_LF2Inv, _L2Inv]]]


    Returns
    -------
    BasisConfig[FundamentalMomentumBasis[_LF0Inv], FundamentalMomentumBasis[_LF1Inv], FundamentalMomentumBasis[_LF2Inv]]
    """
    return (
        basis_as_fundamental_momentum_basis(config[0]),
        basis_as_fundamental_momentum_basis(config[1]),
        basis_as_fundamental_momentum_basis(config[2]),
    )


def basis_config_as_fundamental_position_basis_config(
    config: BasisConfig[
        BasisLike[_LF0Inv, _L0Inv],
        BasisLike[_LF1Inv, _L1Inv],
        BasisLike[_LF2Inv, _L2Inv],
    ]
) -> BasisConfig[
    FundamentalPositionBasis[_LF0Inv],
    FundamentalPositionBasis[_LF1Inv],
    FundamentalPositionBasis[_LF2Inv],
]:
    """
    Get the fundamental postion basis for a given basis.

    Parameters
    ----------
    self : BasisConfigUtil[tuple[BasisLike[_LF0Inv, _L0Inv], BasisLike[_LF1Inv, _L1Inv], BasisLike[_LF2Inv, _L2Inv]]]

    Returns
    -------
    BasisConfig[FundamentalPositionBasis[_LF0Inv], FundamentalPositionBasis[_LF1Inv], FundamentalPositionBasis[_LF2Inv]]
    """
    return (
        basis_as_fundamental_position_basis(config[0]),
        basis_as_fundamental_position_basis(config[1]),
        basis_as_fundamental_position_basis(config[2]),
    )


def basis_config_as_single_point_basis_config(
    config: BasisConfig[
        BasisLike[_LF0Inv, _L0Inv],
        BasisLike[_LF1Inv, _L1Inv],
        BasisLike[_LF2Inv, _L2Inv],
    ]
) -> BasisConfig[
    BasisLike[Literal[1], Literal[1]],
    BasisLike[Literal[1], Literal[1]],
    BasisLike[Literal[1], Literal[1]],
]:
    """
    Get the fundamental postion basis for a given basis.

    Parameters
    ----------
    self : BasisConfigUtil[tuple[BasisLike[_LF0Inv, _L0Inv], BasisLike[_LF1Inv, _L1Inv], BasisLike[_LF2Inv, _L2Inv]]]

    Returns
    -------
    BasisConfig[FundamentalPositionBasis[_LF0Inv], FundamentalPositionBasis[_LF1Inv], FundamentalPositionBasis[_LF2Inv]]
    """
    return (
        basis_as_single_point_basis(config[0]),
        basis_as_single_point_basis(config[1]),
        basis_as_single_point_basis(config[2]),
    )


def _get_rotation_matrix(
    vector: BasisVector, direction: BasisVector | None = None
) -> np.ndarray[tuple[Literal[3], Literal[3]], np.dtype[np.float_]]:
    # From http://www.j3d.org/matrix_faq/matrfaq_latest.html#Q38
    unit = (
        np.array([0.0, 0, 1])
        if direction is None
        else direction.copy() / np.linalg.norm(direction)
    )
    # Normalize vector length
    vector = vector.copy() / np.linalg.norm(vector)

    # Get axis
    uvw = np.cross(vector, unit)

    # compute trig values - no need to go through arccos and back
    rcos: np.float_ = np.dot(vector, unit)
    rsin: np.float_ = np.linalg.norm(uvw)

    # normalize and unpack axis
    if not np.isclose(rsin, 0):
        uvw /= rsin
    u, v, w = uvw

    # Compute rotation matrix - re-expressed to show structure
    return (  # type: ignore[no-any-return]
        rcos * np.eye(3)
        + rsin * np.array([[0, -w, v], [w, 0, -u], [-v, u, 0]])
        + (1.0 - rcos) * uvw[:, None] * uvw[None, :]
    )


def get_rotated_basis_config(
    basis: BasisConfigUtil[BasisConfig[_BX0Inv, _BX1Inv, _BX2Inv]],
    axis: Literal[0, 1, 2, -1, -2, -3] = 0,
    direction: BasisVector | None = None,
) -> BasisConfig[_BX0Inv, _BX1Inv, _BX2Inv]:
    """
    Get the basis, rotated such that axis is along the basis vector direction.

    Parameters
    ----------
    axis : Literal[0, 1, 2, -1, -2, -3], optional
        axis to point along the basis vector direction, by default 0
    direction : BasisVector | None, optional
        basis vector to point along, by default [0,0,1]

    Returns
    -------
    BasisConfig[_BX0Cov, _BX1Cov, _BX2Cov]
        _description_
    """
    matrix = _get_rotation_matrix(basis.config[axis].delta_x, direction)
    return (
        get_rotated_basis(basis.config[0], matrix),
        get_rotated_basis(basis.config[1], matrix),
        get_rotated_basis(basis.config[2], matrix),
    )
