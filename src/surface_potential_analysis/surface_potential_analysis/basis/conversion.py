from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

import numpy as np

from surface_potential_analysis.axis.axis import (
    FundamentalMomentumAxis,
    FundamentalMomentumAxis1d,
    FundamentalMomentumAxis2d,
    FundamentalMomentumAxis3d,
    FundamentalPositionAxis,
    FundamentalPositionAxis1d,
    FundamentalPositionAxis2d,
    FundamentalPositionAxis3d,
    MomentumAxis,
)
from surface_potential_analysis.axis.conversion import (
    axis_as_fundamental_momentum_axis,
    axis_as_fundamental_position_axis,
    axis_as_n_point_axis,
    axis_as_single_point_axis,
)
from surface_potential_analysis.util.interpolation import pad_ft_points

from .util import BasisUtil

if TYPE_CHECKING:
    from surface_potential_analysis.axis.axis_like import (
        AxisLike,
        AxisLike1d,
        AxisLike2d,
        AxisLike3d,
    )
    from surface_potential_analysis.basis.basis import Basis, Basis1d, Basis2d

    from .basis import Basis3d

    _A0Inv = TypeVar("_A0Inv", bound=AxisLike[Any, Any, Any])
    _A1Inv = TypeVar("_A1Inv", bound=AxisLike[Any, Any, Any])

    _B0Inv = TypeVar("_B0Inv", bound=Basis[Any])
    _B1Inv = TypeVar("_B1Inv", bound=Basis[Any])
    _B2Inv = TypeVar("_B2Inv", bound=Basis[Any])
    _B3Inv = TypeVar("_B3Inv", bound=Basis[Any])
    _B1d0Inv = TypeVar("_B1d0Inv", bound=Basis1d[Any])
    _B1d1Inv = TypeVar("_B1d1Inv", bound=Basis1d[Any])
    _B1d2Inv = TypeVar("_B1d2Inv", bound=Basis1d[Any])
    _B1d3Inv = TypeVar("_B1d3Inv", bound=Basis1d[Any])
    _B2d0Inv = TypeVar("_B2d1Inv", bound=Basis2d[Any, Any])
    _B2d1Inv = TypeVar("_B2d0Inv", bound=Basis2d[Any, Any])
    _B2d2Inv = TypeVar("_B2d2Inv", bound=Basis2d[Any, Any])
    _B2d3Inv = TypeVar("_B2d3Inv", bound=Basis2d[Any, Any])
    _B3d0Inv = TypeVar("_B3d0Inv", bound=Basis3d[Any, Any, Any])
    _B3d1Inv = TypeVar("_B3d1Inv", bound=Basis3d[Any, Any, Any])
    _B3d2Inv = TypeVar("_B3d2Inv", bound=Basis3d[Any, Any, Any])
    _B3d3Inv = TypeVar("_B3d3Inv", bound=Basis3d[Any, Any, Any])
    _S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])
    _NDInv = TypeVar("_NDInv", bound=int)


def _convert_vector_along_axis_simple(
    vector: np.ndarray[_S0Inv, np.dtype[np.complex_ | np.float_]],
    initial_axis: _A0Inv,
    final_axis: _A1Inv,
    axis: int,
) -> np.ndarray[Any, np.dtype[np.complex_]]:
    fundamental = initial_axis.__into_fundamental__(vector, axis)
    return final_axis.__from_fundamental__(fundamental, axis)


def _convert_vector_along_axis(
    vector: np.ndarray[_S0Inv, np.dtype[np.complex_ | np.float_]],
    initial_axis: _A0Inv,
    final_axis: _A1Inv,
    axis: int,
) -> np.ndarray[Any, np.dtype[np.complex_]]:
    # Small speedup here, and prevents imprecision of fft followed by ifft
    # And two pad_ft_points
    if isinstance(initial_axis, MomentumAxis) and isinstance(final_axis, MomentumAxis):
        padded = pad_ft_points(vector, s=(final_axis.n,), axes=(axis,))
        return padded.astype(np.complex_)  # type: ignore[no-any-return]
    return _convert_vector_along_axis_simple(vector, initial_axis, final_axis, axis)


@overload
def convert_vector(
    vector: np.ndarray[_S0Inv, np.dtype[np.complex_] | np.dtype[np.float_]],
    initial_basis: _B1d0Inv,
    final_basis: _B1d1Inv,
    axis: int = -1,
) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
    ...


@overload
def convert_vector(
    vector: np.ndarray[_S0Inv, np.dtype[np.complex_] | np.dtype[np.float_]],
    initial_basis: _B2d0Inv,
    final_basis: _B2d1Inv,
    axis: int = -1,
) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
    ...


@overload
def convert_vector(
    vector: np.ndarray[_S0Inv, np.dtype[np.complex_] | np.dtype[np.float_]],
    initial_basis: _B3d0Inv,
    final_basis: _B3d1Inv,
    axis: int = -1,
) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
    ...


@overload
def convert_vector(
    vector: np.ndarray[_S0Inv, np.dtype[np.complex_] | np.dtype[np.float_]],
    initial_basis: _B0Inv,
    final_basis: _B1Inv,
    axis: int = -1,
) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
    ...


def convert_vector(
    vector: np.ndarray[_S0Inv, np.dtype[np.complex_] | np.dtype[np.float_]],
    initial_basis: _B0Inv,
    final_basis: _B1Inv,
    axis: int = -1,
) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
    """
    Convert a vector, expressed in terms of the given basis from_config in the basis to_config.

    Parameters
    ----------
    vector : np.ndarray[tuple[int], np.dtype[np.complex_] | np.dtype[np.float_]]
        the vector to convert
    from_config : _B3d0Inv
    to_config : _B3d1Inv
    axis : int, optional
        axis along which to convert, by default -1

    Returns
    -------
    np.ndarray[tuple[int], np.dtype[np.complex_]]
    """
    util = BasisUtil(initial_basis)
    swapped = vector.swapaxes(axis, 0)
    stacked = swapped.reshape(*util.shape, *swapped.shape[1:])
    for ax, (initial, final) in enumerate(zip(initial_basis, final_basis, strict=True)):
        stacked = _convert_vector_along_axis(stacked, initial, final, ax)
    return stacked.astype(np.complex_).reshape(-1, *swapped.shape[1:]).swapaxes(axis, 0)  # type: ignore[no-any-return]


@overload
def convert_dual_vector(
    co_vector: np.ndarray[_S0Inv, np.dtype[np.complex_] | np.dtype[np.float_]],
    initial_basis: _B1d0Inv,
    final_basis: _B1d1Inv,
    axis: int = -1,
) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
    ...


@overload
def convert_dual_vector(
    co_vector: np.ndarray[_S0Inv, np.dtype[np.complex_] | np.dtype[np.float_]],
    initial_basis: _B2d0Inv,
    final_basis: _B2d1Inv,
    axis: int = -1,
) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
    ...


@overload
def convert_dual_vector(
    co_vector: np.ndarray[_S0Inv, np.dtype[np.complex_] | np.dtype[np.float_]],
    initial_basis: _B3d0Inv,
    final_basis: _B3d1Inv,
    axis: int = -1,
) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
    ...


@overload
def convert_dual_vector(
    co_vector: np.ndarray[_S0Inv, np.dtype[np.complex_] | np.dtype[np.float_]],
    initial_basis: _B0Inv,
    final_basis: _B1Inv,
    axis: int = -1,
) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
    ...


def convert_dual_vector(
    co_vector: np.ndarray[_S0Inv, np.dtype[np.complex_] | np.dtype[np.float_]],
    initial_basis: _B0Inv,
    final_basis: _B1Inv,
    axis: int = -1,
) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
    """
    Convert a co_vector, expressed in terms of the given basis from_config in the basis to_config.

    Parameters
    ----------
    co_vector : np.ndarray[tuple[int], np.dtype[np.complex_]]
        the vector to convert
    from_config : _B3d0Inv
    to_config : _B3d1Inv
    axis : int, optional
        axis along which to convert, by default -1

    Returns
    -------
    np.ndarray[tuple[int], np.dtype[np.complex_]]
    """
    return np.conj(convert_vector(np.conj(co_vector), initial_basis, final_basis, axis))  # type: ignore[no-any-return]


@overload
def convert_matrix(
    matrix: np.ndarray[tuple[int, int], np.dtype[np.complex_] | np.dtype[np.float_]],
    initial_basis: _B1d0Inv,
    final_basis: _B1d1Inv,
    initial_dual_basis: _B1d2Inv,
    final_dual_basis: _B1d3Inv,
) -> np.ndarray[tuple[int, int], np.dtype[np.complex_]]:
    ...


@overload
def convert_matrix(
    matrix: np.ndarray[tuple[int, int], np.dtype[np.complex_] | np.dtype[np.float_]],
    initial_basis: _B2d0Inv,
    final_basis: _B2d1Inv,
    initial_dual_basis: _B2d2Inv,
    final_dual_basis: _B2d3Inv,
) -> np.ndarray[tuple[int, int], np.dtype[np.complex_]]:
    ...


@overload
def convert_matrix(
    matrix: np.ndarray[tuple[int, int], np.dtype[np.complex_] | np.dtype[np.float_]],
    initial_basis: _B3d0Inv,
    final_basis: _B3d1Inv,
    initial_dual_basis: _B3d2Inv,
    final_dual_basis: _B3d3Inv,
) -> np.ndarray[tuple[int, int], np.dtype[np.complex_]]:
    ...


@overload
def convert_matrix(
    matrix: np.ndarray[tuple[int, int], np.dtype[np.complex_] | np.dtype[np.float_]],
    initial_basis: _B0Inv,
    final_basis: _B1Inv,
    initial_dual_basis: _B2Inv,
    final_dual_basis: _B3Inv,
) -> np.ndarray[tuple[int, int], np.dtype[np.complex_]]:
    ...


def convert_matrix(
    matrix: np.ndarray[tuple[int, int], np.dtype[np.complex_] | np.dtype[np.float_]],
    initial_basis: _B0Inv,
    final_basis: _B1Inv,
    initial_dual_basis: _B2Inv,
    final_dual_basis: _B3Inv,
) -> np.ndarray[tuple[int, int], np.dtype[np.complex_]]:
    """
    Convert a matrix from initial_basis to final_basis.

    Parameters
    ----------
    matrix : np.ndarray[tuple[int, int], np.dtype[np.complex_]]
    initial_basis : _B3d0Inv
    final_basis : _B3d1Inv

    Returns
    -------
    np.ndarray[tuple[int, int], np.dtype[np.complex_]]
    """
    converted = convert_vector(matrix, initial_basis, final_basis, axis=0)
    return convert_dual_vector(converted, initial_dual_basis, final_dual_basis, axis=1)  # type: ignore[return-value]


_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)

_LF0Inv = TypeVar("_LF0Inv", bound=int)
_LF1Inv = TypeVar("_LF1Inv", bound=int)
_LF2Inv = TypeVar("_LF2Inv", bound=int)


@overload
def basis_as_fundamental_momentum_basis(
    basis: Basis1d[AxisLike1d[_LF0Inv, _L0Inv]]
) -> Basis1d[FundamentalMomentumAxis1d[_LF0Inv]]:
    ...


@overload
def basis_as_fundamental_momentum_basis(
    basis: Basis2d[AxisLike2d[_LF0Inv, _L0Inv], AxisLike2d[_LF1Inv, _L1Inv]]
) -> Basis2d[FundamentalMomentumAxis2d[_LF0Inv], FundamentalMomentumAxis2d[_LF1Inv]]:
    ...


@overload
def basis_as_fundamental_momentum_basis(
    basis: Basis3d[
        AxisLike3d[_LF0Inv, _L0Inv],
        AxisLike3d[_LF1Inv, _L1Inv],
        AxisLike3d[_LF2Inv, _L2Inv],
    ]
) -> Basis3d[
    FundamentalMomentumAxis3d[_LF0Inv],
    FundamentalMomentumAxis3d[_LF1Inv],
    FundamentalMomentumAxis3d[_LF2Inv],
]:
    ...


@overload
def basis_as_fundamental_momentum_basis(
    basis: _B0Inv,
) -> tuple[FundamentalMomentumAxis[Any, Any], ...]:
    ...


def basis_as_fundamental_momentum_basis(
    basis: _B0Inv,
) -> tuple[FundamentalMomentumAxis[Any, Any], ...]:
    """
    Get the fundamental momentum basis for a given basis.

    Parameters
    ----------
    basis : _B0Inv

    Returns
    -------
    tuple[FundamentalMomentumAxis[Any, Any], ...]
    """
    return tuple(axis_as_fundamental_momentum_axis(axis) for axis in basis)


@overload
def basis_as_fundamental_position_basis(
    basis: Basis1d[AxisLike1d[_LF0Inv, _L0Inv]]
) -> Basis1d[FundamentalPositionAxis1d[_LF0Inv]]:
    ...


@overload
def basis_as_fundamental_position_basis(
    basis: Basis2d[AxisLike2d[_LF0Inv, _L0Inv], AxisLike2d[_LF1Inv, _L1Inv]]
) -> Basis2d[FundamentalPositionAxis2d[_LF0Inv], FundamentalPositionAxis2d[_LF1Inv]]:
    ...


@overload
def basis_as_fundamental_position_basis(
    basis: Basis3d[
        AxisLike3d[_LF0Inv, _L0Inv],
        AxisLike3d[_LF1Inv, _L1Inv],
        AxisLike3d[_LF2Inv, _L2Inv],
    ]
) -> Basis3d[
    FundamentalPositionAxis3d[_LF0Inv],
    FundamentalPositionAxis3d[_LF1Inv],
    FundamentalPositionAxis3d[_LF2Inv],
]:
    ...


@overload
def basis_as_fundamental_position_basis(
    basis: _B0Inv,
) -> tuple[FundamentalPositionAxis[Any, Any], ...]:
    ...


def basis_as_fundamental_position_basis(
    basis: _B0Inv,
) -> tuple[FundamentalPositionAxis[Any, Any], ...]:
    """
    Get the fundamental position basis for a given basis.

    Parameters
    ----------
    self : BasisConfigUtil[tuple[BasisLike[_LF0Inv, _L0Inv], BasisLike[_LF1Inv, _L1Inv], BasisLike[_LF2Inv, _L2Inv]]]

    Returns
    -------
    Basis3d[FundamentalPositionBasis[_LF0Inv], FundamentalPositionBasis[_LF1Inv], FundamentalPositionBasis[_LF2Inv]]
    """
    return tuple(axis_as_fundamental_position_axis(axis) for axis in basis)


def basis_as_fundamental_with_shape(
    basis: Basis[_NDInv],
    shape: tuple[int, ...],
) -> Basis[_NDInv]:
    """
    Given a basis get a fundamental position basis with the given shape.

    Parameters
    ----------
    basis : Basis[_NDInv]
    shape : tuple[int, ...]

    Returns
    -------
    Basis[_NDInv]
    """
    return tuple(
        axis_as_n_point_axis(ax, n=n) for (ax, n) in zip(basis, shape, strict=True)
    )


def basis3d_as_single_point_basis(
    basis: Basis3d[
        AxisLike3d[_LF0Inv, _L0Inv],
        AxisLike3d[_LF1Inv, _L1Inv],
        AxisLike3d[_LF2Inv, _L2Inv],
    ]
) -> Basis3d[
    AxisLike3d[Literal[1], Literal[1]],
    AxisLike3d[Literal[1], Literal[1]],
    AxisLike3d[Literal[1], Literal[1]],
]:
    """
    Get the fundamental single point basis for a given basis.

    Parameters
    ----------
    self : BasisConfigUtil[tuple[BasisLike[_LF0Inv, _L0Inv], BasisLike[_LF1Inv, _L1Inv], BasisLike[_LF2Inv, _L2Inv]]]

    Returns
    -------
    Basis3d[FundamentalPositionBasis[_LF0Inv], FundamentalPositionBasis[_LF1Inv], FundamentalPositionBasis[_LF2Inv]]
    """
    return (
        axis_as_single_point_axis(basis[0]),
        axis_as_single_point_axis(basis[1]),
        axis_as_single_point_axis(basis[2]),
    )
