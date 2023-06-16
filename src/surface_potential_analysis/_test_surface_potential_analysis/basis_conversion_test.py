from __future__ import annotations

import unittest
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from _test_surface_potential_analysis.utils import get_random_explicit_axis
from surface_potential_analysis.axis.axis import (
    FundamentalMomentumAxis,
)
from surface_potential_analysis.axis.util import AxisUtil
from surface_potential_analysis.basis.conversion import (
    basis_as_fundamental_position_basis,
    convert_vector,
)
from surface_potential_analysis.basis.util import BasisUtil

if TYPE_CHECKING:
    from surface_potential_analysis.axis.axis_like import AxisLike
    from surface_potential_analysis.basis.basis import Basis

    _B0Inv = TypeVar("_B0Inv", bound=Basis[Any])
    _B1Inv = TypeVar("_B1Inv", bound=Basis[Any])
    _S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])
    _NDInv = TypeVar("_NDInv", bound=int)

    _A0Inv = TypeVar("_A0Inv", bound=AxisLike[Any, Any, Any])
    _A1Inv = TypeVar("_A1Inv", bound=AxisLike[Any, Any, Any])

    _N0Inv = TypeVar("_N0Inv", bound=int)
    _N1Inv = TypeVar("_N1Inv", bound=int)

    _NF0Inv = TypeVar("_NF0Inv", bound=int)
    _NF1Inv = TypeVar("_NF1Inv", bound=int)

rng = np.random.default_rng()


def get_axis_conversion_matrix(
    axis_0: AxisLike[_N0Inv, _NF0Inv, _NDInv], axis_1: AxisLike[_N1Inv, _NF1Inv, _NDInv]
) -> np.ndarray[tuple[_NF0Inv, _NF1Inv], np.dtype[np.complex_]]:
    """
    Get the matrix to convert one set of axis axes into another.

    Parameters
    ----------
    axis_0 : AxisLike[_N0Inv, _NF0Inv]
    axis_1 : AxisLike[_N1Inv, _NF1Inv]

    Returns
    -------
    np.ndarray[tuple[_NF0Inv, _NF1Inv], np.dtype[np.complex_]]
    """
    vectors_0 = AxisUtil(axis_0).vectors
    vectors_1 = AxisUtil(axis_1).vectors
    return np.dot(vectors_0, np.conj(vectors_1).T)  # type: ignore[no-any-return]


def _convert_vector_along_axis(
    vector: np.ndarray[_S0Inv, np.dtype[np.complex_]],
    initial_axis: _A0Inv,
    final_axis: _A1Inv,
    axis: int,
) -> np.ndarray[Any, np.dtype[np.complex_]]:
    matrix = get_axis_conversion_matrix(initial_axis, final_axis)
    return np.moveaxis(np.tensordot(vector, matrix, axes=([axis], [0])), -1, axis)  # type: ignore[no-any-return]


def convert_vector_simple(
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
    stacked = swapped.astype(np.complex_, copy=False).reshape(
        *swapped.shape[:-1], *util.shape
    )
    last_axis = swapped.ndim - 1
    for convert_axis, initial, final in zip(
        range(last_axis, stacked.ndim), initial_basis, final_basis, strict=True
    ):
        stacked = _convert_vector_along_axis(stacked, initial, final, convert_axis)

    return stacked.reshape(*swapped.shape[:-1], -1).swapaxes(axis, -1)  # type: ignore[no-any-return]


class BasisConversionTest(unittest.TestCase):
    def test_as_explicit_position_basis_momentum_normalization(self) -> None:
        fundamental_n = rng.integers(2, 5)
        n = rng.integers(1, fundamental_n)

        basis = (
            AxisUtil(get_random_explicit_axis(1, fundamental_n=fundamental_n, n=n)),
        )

        np.testing.assert_array_almost_equal(
            np.linalg.norm(basis[0].vectors, axis=1), np.ones(n)
        )

        actual = convert_vector(
            basis[0].vectors, basis_as_fundamental_position_basis(basis), basis
        )
        expected = np.eye(n)
        np.testing.assert_almost_equal(actual, expected)
        actual = convert_vector(
            basis[0].vectors, basis_as_fundamental_position_basis(basis), basis
        )
        expected = convert_vector_simple(
            basis[0].vectors, basis_as_fundamental_position_basis(basis), basis
        )
        np.testing.assert_almost_equal(actual, expected)

        actual = convert_vector(
            np.eye(fundamental_n), basis_as_fundamental_position_basis(basis), basis
        )
        expected = basis[0].vectors.T
        np.testing.assert_almost_equal(actual, expected)

        actual = convert_vector(
            np.eye(fundamental_n), basis_as_fundamental_position_basis(basis), basis
        )
        expected = convert_vector_simple(
            np.eye(fundamental_n), basis_as_fundamental_position_basis(basis), basis
        )
        np.testing.assert_almost_equal(actual, expected)

    def test_as_explicit_position_basis_momentum(self) -> None:
        n = rng.integers(5, 10)

        basis = (FundamentalMomentumAxis(np.array([1]), n),)

        np.testing.assert_array_almost_equal(
            np.linalg.norm(basis[0].vectors, axis=1), np.ones(n)
        )
        np.testing.assert_array_almost_equal(
            basis[0].vectors,
            np.exp(
                (1j * 2 * np.pi)
                * np.arange(n)[:, np.newaxis]
                * np.linspace(0, 1, n, endpoint=False)[np.newaxis, :]
            )
            / np.sqrt(n),
        )

        np.testing.assert_array_almost_equal(
            convert_vector(
                np.eye(n), basis, basis_as_fundamental_position_basis(basis)
            ),
            np.exp(
                (1j * 2 * np.pi)
                * np.arange(n)[:, np.newaxis]
                * np.linspace(0, 1, n, endpoint=False)[np.newaxis, :]
            )
            / np.sqrt(n),
        )

        np.testing.assert_array_almost_equal(
            convert_vector(
                np.eye(n), basis, basis_as_fundamental_position_basis(basis)
            ),
            convert_vector_simple(
                np.eye(n), basis, basis_as_fundamental_position_basis(basis)
            ),
        )

    def test_get_basis_conversion_matrix_diagonal(self) -> None:
        fundamental_n = rng.integers(2, 5)
        n = rng.integers(1, fundamental_n)

        basis_0 = AxisUtil(
            get_random_explicit_axis(1, fundamental_n=fundamental_n, n=n)
        )
        np.testing.assert_array_equal(basis_0.vectors.shape, (n, fundamental_n))

        matrix = get_axis_conversion_matrix(basis_0, basis_0)
        np.testing.assert_array_almost_equal(matrix, np.eye(n, n))

        basis_1 = get_random_explicit_axis(1, fundamental_n=fundamental_n, n=n)

        matrix = get_axis_conversion_matrix(basis_1, basis_1)
        np.testing.assert_array_almost_equal(matrix, np.eye(n, n))

    def test_basis_conversion_matrix_position(self) -> None:
        fundamental_n = rng.integers(2, 5)
        n = rng.integers(1, fundamental_n)
        basis_0 = get_random_explicit_axis(1, fundamental_n=fundamental_n, n=n)

        matrix = get_axis_conversion_matrix(basis_0, basis_0)

        np.testing.assert_array_almost_equal(matrix, np.eye(n, n))

        basis_1 = get_random_explicit_axis(1, fundamental_n=fundamental_n, n=n)

        matrix = get_axis_conversion_matrix(basis_1, basis_1)

        np.testing.assert_array_almost_equal(matrix, np.eye(n, n))
