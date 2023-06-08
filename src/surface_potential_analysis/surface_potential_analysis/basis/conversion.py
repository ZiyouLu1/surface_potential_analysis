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
    get_axis_conversion_matrix,
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
    _B1d0Inv = TypeVar("_B1d0Inv", bound=Basis1d[Any])
    _B1d1Inv = TypeVar("_B1d1Inv", bound=Basis1d[Any])
    _B2d0Inv = TypeVar("_B2d0Inv", bound=Basis2d[Any, Any])
    _B2d1Inv = TypeVar("_B2d1Inv", bound=Basis2d[Any, Any])
    _B3d0Inv = TypeVar("_B3d0Inv", bound=Basis3d[Any, Any, Any])
    _B3d1Inv = TypeVar("_B3d1Inv", bound=Basis3d[Any, Any, Any])
    _S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])
    _NDInv = TypeVar("_NDInv", bound=int)


def _convert_vector_along_axis_simple(
    vector: np.ndarray[_S0Inv, np.dtype[np.complex_]],
    initial_axis: _A0Inv,
    final_axis: _A1Inv,
    axis: int,
) -> np.ndarray[Any, np.dtype[np.complex_]]:
    matrix = get_axis_conversion_matrix(initial_axis, final_axis)
    return np.moveaxis(np.tensordot(vector, matrix, axes=([axis], [0])), -1, axis)  # type: ignore[no-any-return]


def _convert_vector_along_axis(
    vector: np.ndarray[_S0Inv, np.dtype[np.complex_]],
    initial_axis: _A0Inv,
    final_axis: _A1Inv,
    axis: int,
) -> np.ndarray[Any, np.dtype[np.complex_]]:
    if isinstance(initial_axis, FundamentalPositionAxis) and isinstance(
        final_axis, FundamentalPositionAxis
    ):
        return vector
    if isinstance(initial_axis, FundamentalPositionAxis) and isinstance(
        final_axis, MomentumAxis
    ):
        transformed = np.fft.fft(vector, axis=axis, norm="ortho")
        return pad_ft_points(transformed, s=(final_axis.n,), axes=(0,))
    if isinstance(initial_axis, MomentumAxis) and isinstance(
        final_axis, FundamentalPositionAxis
    ):
        padded = pad_ft_points(vector, s=(final_axis.n,), axes=(axis,))
        return np.fft.ifft(padded, axis=axis, norm="ortho")  # type: ignore[no-any-return]
    if isinstance(initial_axis, MomentumAxis) and isinstance(final_axis, MomentumAxis):
        return pad_ft_points(vector, s=(final_axis.n,), axes=(axis,))
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
    swapped = vector.swapaxes(axis, -1)
    stacked = swapped.astype(np.complex_).reshape(*swapped.shape[:-1], *util.shape)
    last_axis = swapped.ndim - 1
    for convert_axis, initial, final in zip(
        range(last_axis, stacked.ndim), initial_basis, final_basis, strict=True
    ):
        stacked = _convert_vector_along_axis(stacked, initial, final, convert_axis)

    return stacked.reshape(*swapped.shape[:-1], -1).swapaxes(axis, -1)  # type: ignore[no-any-return]


@overload
def convert_co_vector(
    co_vector: np.ndarray[_S0Inv, np.dtype[np.complex_] | np.dtype[np.float_]],
    initial_basis: _B1d0Inv,
    final_basis: _B1d1Inv,
    axis: int = -1,
) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
    ...


@overload
def convert_co_vector(
    co_vector: np.ndarray[_S0Inv, np.dtype[np.complex_] | np.dtype[np.float_]],
    initial_basis: _B2d0Inv,
    final_basis: _B2d1Inv,
    axis: int = -1,
) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
    ...


@overload
def convert_co_vector(
    co_vector: np.ndarray[_S0Inv, np.dtype[np.complex_] | np.dtype[np.float_]],
    initial_basis: _B3d0Inv,
    final_basis: _B3d1Inv,
    axis: int = -1,
) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
    ...


@overload
def convert_co_vector(
    co_vector: np.ndarray[_S0Inv, np.dtype[np.complex_] | np.dtype[np.float_]],
    initial_basis: _B0Inv,
    final_basis: _B1Inv,
    axis: int = -1,
) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
    ...


def convert_co_vector(
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
) -> np.ndarray[tuple[int, int], np.dtype[np.complex_]]:
    ...


@overload
def convert_matrix(
    matrix: np.ndarray[tuple[int, int], np.dtype[np.complex_] | np.dtype[np.float_]],
    initial_basis: _B2d0Inv,
    final_basis: _B2d1Inv,
) -> np.ndarray[tuple[int, int], np.dtype[np.complex_]]:
    ...


@overload
def convert_matrix(
    matrix: np.ndarray[tuple[int, int], np.dtype[np.complex_] | np.dtype[np.float_]],
    initial_basis: _B3d0Inv,
    final_basis: _B3d1Inv,
) -> np.ndarray[tuple[int, int], np.dtype[np.complex_]]:
    ...


@overload
def convert_matrix(
    matrix: np.ndarray[tuple[int, int], np.dtype[np.complex_] | np.dtype[np.float_]],
    initial_basis: _B0Inv,
    final_basis: _B1Inv,
) -> np.ndarray[tuple[int, int], np.dtype[np.complex_]]:
    ...


def convert_matrix(
    matrix: np.ndarray[tuple[int, int], np.dtype[np.complex_] | np.dtype[np.float_]],
    initial_basis: _B0Inv,
    final_basis: _B1Inv,
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
    return convert_co_vector(converted, initial_basis, final_basis, axis=1)  # type: ignore[return-value]


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
