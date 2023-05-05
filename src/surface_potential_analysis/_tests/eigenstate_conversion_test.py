from __future__ import annotations

import unittest
from typing import TYPE_CHECKING, Any, Generic, TypedDict, TypeVar

import hamiltonian_generator
import numpy as np
from scipy.constants import hbar

from _tests.utils import get_random_explicit_basis
from surface_potential_analysis.basis.basis import (
    Basis,
    ExplicitBasis,
    FundamentalBasis,
    MomentumBasis,
    PositionBasis,
    TruncatedBasis,
    as_fundamental_basis,
)
from surface_potential_analysis.basis_config.basis_config import (
    BasisConfig,
    BasisConfigUtil,
    PositionBasisConfigUtil,
)
from surface_potential_analysis.basis_config.sho_basis import (
    SHOBasisConfig,
    infinate_sho_basis_from_config,
)
from surface_potential_analysis.eigenstate.conversion import (
    convert_eigenstate_to_basis,
)
from surface_potential_analysis.interpolation import pad_ft_points

if TYPE_CHECKING:
    from surface_potential_analysis.eigenstate.eigenstate import (
        Eigenstate,
        EigenstateWithBasis,
        PositionBasisEigenstate,
    )

_rng = np.random.default_rng()

_BC0Cov = TypeVar("_BC0Cov", bound=BasisConfig[Any, Any, Any], covariant=True)
_BC0Inv = TypeVar("_BC0Inv", bound=BasisConfig[Any, Any, Any])

_BX0Cov = TypeVar("_BX0Cov", bound=Basis[Any, Any], covariant=True)
_BX1Cov = TypeVar("_BX1Cov", bound=Basis[Any, Any], covariant=True)
_BX2Cov = TypeVar("_BX2Cov", bound=Basis[Any, Any], covariant=True)

_BX0Inv = TypeVar("_BX0Inv", bound=Basis[Any, Any])
_BX1Inv = TypeVar("_BX1Inv", bound=Basis[Any, Any])

_L0Inv = TypeVar("_L0Inv", bound=int, covariant=True)
_L1Inv = TypeVar("_L1Inv", bound=int, covariant=True)
_L2Inv = TypeVar("_L2Inv", bound=int, covariant=True)

_LF0Inv = TypeVar("_LF0Inv", bound=int)
_LF1Inv = TypeVar("_LF1Inv", bound=int)
_LF2Inv = TypeVar("_LF2Inv", bound=int)

_PInv = TypeVar("_PInv", bound=FundamentalBasis[Any])


class _StackedEigenstate(TypedDict, Generic[_BC0Cov]):
    basis: _BC0Cov
    vector: np.ndarray[tuple[int, int, int], np.dtype[np.complex_]]


StackedEigenstateWithBasis = _StackedEigenstate[BasisConfig[_BX0Cov, _BX1Cov, _BX2Cov]]


def _stack_eigenstate(state: Eigenstate[_BC0Inv]) -> _StackedEigenstate[_BC0Inv]:
    util = BasisConfigUtil(state["basis"])
    return {"basis": state["basis"], "vector": state["vector"].reshape(util.shape)}


def _flatten_eigenstate(state: _StackedEigenstate[_BC0Inv]) -> Eigenstate[_BC0Inv]:
    return {"basis": state["basis"], "vector": state["vector"].reshape(-1)}


def _convert_explicit_basis_x2_to_position(
    eigenstate: StackedEigenstateWithBasis[_BX0Inv, _BX1Inv, ExplicitBasis[Any, _PInv]]
) -> StackedEigenstateWithBasis[_BX0Inv, _BX1Inv, _PInv]:
    vector = np.sum(
        eigenstate["vector"][:, :, :, np.newaxis]
        * eigenstate["basis"][2]["vectors"][np.newaxis, np.newaxis, :, :],
        axis=2,
    )
    return {
        "basis": (
            eigenstate["basis"][0],
            eigenstate["basis"][1],
            eigenstate["basis"][2]["parent"],
        ),
        "vector": vector,
    }

def _convert_momentum_basis_x01_to_position(
    eigenstate: StackedEigenstateWithBasis[
        TruncatedBasis[Any, MomentumBasis[_L0Inv]] | MomentumBasis[_L0Inv],
        TruncatedBasis[Any, MomentumBasis[_L1Inv]] | MomentumBasis[_L1Inv],
        _BX0Inv,
    ]
    | StackedEigenstateWithBasis[
        MomentumBasis[_L0Inv],
        MomentumBasis[_L1Inv],
        _BX0Inv,
    ]
    | StackedEigenstateWithBasis[
        TruncatedBasis[Any, MomentumBasis[_L0Inv]],
        TruncatedBasis[Any, MomentumBasis[_L1Inv]],
        _BX0Inv,
    ]
) -> StackedEigenstateWithBasis[PositionBasis[_L0Inv], PositionBasis[_L1Inv], _BX0Inv]:
    util = BasisConfigUtil(eigenstate["basis"])
    padded = pad_ft_points(
        eigenstate["vector"],  # type: ignore[arg-type]
        s=[util.fundamental_n0, util.fundamental_n1],
        axes=(0, 1),
    )
    transformed = np.fft.ifftn(padded, axes=(0, 1), norm="ortho")
    return {
        "basis": (
            {
                "_type": "position",
                "delta_x": util.delta_x0,
                "n": util.fundamental_n0,  # type: ignore[typeddict-item]
            },
            {
                "_type": "position",
                "delta_x": util.delta_x1,
                "n": util.fundamental_n1,  # type: ignore[typeddict-item]
            },
            eigenstate["basis"][2],
        ),
        "vector": transformed,
    }



def _convert_sho_eigenstate_to_position_basis(
    eigenstate: EigenstateWithBasis[
        TruncatedBasis[_L0Inv, MomentumBasis[_LF0Inv]] | MomentumBasis[_LF0Inv],
        TruncatedBasis[_L1Inv, MomentumBasis[_LF1Inv]] | MomentumBasis[_LF1Inv],
        ExplicitBasis[_L2Inv, PositionBasis[_LF2Inv]],
    ]
    | EigenstateWithBasis[
        MomentumBasis[_LF0Inv],
        MomentumBasis[_LF1Inv],
        ExplicitBasis[_L2Inv, PositionBasis[_LF2Inv]],
    ]
    | EigenstateWithBasis[
        TruncatedBasis[_L0Inv, MomentumBasis[_LF0Inv]],
        TruncatedBasis[_L1Inv, MomentumBasis[_LF1Inv]],
        ExplicitBasis[_L2Inv, PositionBasis[_LF2Inv]],
    ]
) -> PositionBasisEigenstate[_LF0Inv, _LF1Inv, _LF2Inv]:
    """
    Given an eigenstate in sho basis convert it into position basis.

    Parameters
    ----------
    eigenstate : EigenstateWithBasis[ TruncatedBasis[_L0Inv, MomentumBasis[_LF0Inv]]  |  MomentumBasis[_LF0Inv], TruncatedBasis[_L1Inv, MomentumBasis[_LF1Inv]]  |  MomentumBasis[_LF1Inv], ExplicitBasis[_L2Inv, PositionBasis[_LF2Inv]], ]

    Returns
    -------
    PositionBasisEigenstate[_LF0Inv, _LF1Inv, _LF2Inv]
    """
    stacked = _stack_eigenstate(eigenstate)  # type: ignore[arg-type]
    xy_converted = _convert_momentum_basis_x01_to_position(stacked)
    converted = _convert_explicit_basis_x2_to_position(xy_converted)
    return _flatten_eigenstate(converted)


def _get_random_sho_eigenstate(
    resolution: tuple[int, int, int], fundamental_resolution: tuple[int, int, int]
) -> EigenstateWithBasis[
    TruncatedBasis[Any, MomentumBasis[Any]],
    TruncatedBasis[Any, MomentumBasis[Any]],
    ExplicitBasis[int, Any],
]:
    vector = np.array(_rng.random(np.prod(resolution)), dtype=complex)
    vector /= np.linalg.norm(vector)

    x2_basis = get_random_explicit_basis(
        "position", fundamental_n=fundamental_resolution[2], n=resolution[2]
    )
    x2_basis["parent"]["delta_x"] = np.array([0, 0, 20])
    return {
        "basis": (
            {
                "_type": "truncated",
                "n": resolution[0],
                "parent": {
                    "_type": "momentum",
                    "delta_x": np.array([1, 0, 0]),
                    "n": fundamental_resolution[0],
                },
            },
            {
                "_type": "truncated",
                "n": resolution[1],
                "parent": {
                    "_type": "momentum",
                    "delta_x": np.array([0, 1, 0]),
                    "n": fundamental_resolution[1],
                },
            },
            x2_basis,
        ),
        "vector": vector,
    }


class EigenstateConversionTest(unittest.TestCase):
    def test_random_stack_eigenstate(self) -> None:
        basis = PositionBasisConfigUtil.from_resolution((10, 12, 13))
        util = PositionBasisConfigUtil(basis)
        eigenstate: Eigenstate[Any] = {
            "basis": basis,
            "vector": np.array(_rng.random(len(util)), dtype=complex),
        }
        stacked_eigenstate = _stack_eigenstate(eigenstate)

        np.testing.assert_array_equal(
            eigenstate["vector"],
            stacked_eigenstate["vector"][*util.fundamental_nx_points],
        )

        np.testing.assert_array_equal(
            eigenstate["vector"],
            _flatten_eigenstate(stacked_eigenstate)["vector"],
        )

    def test_convert_explicit_basis_x2_to_position_shape(self) -> None:
        eigenstate = _get_random_sho_eigenstate((10, 12, 9), (10, 12, 13))

        eigenstate_stacked = _stack_eigenstate(eigenstate)
        stacked_position = _convert_explicit_basis_x2_to_position(eigenstate_stacked)

        np.testing.assert_array_equal((10, 12, 13), stacked_position["vector"].shape)

        position = _flatten_eigenstate(stacked_position)
        expected = convert_eigenstate_to_basis(eigenstate, position["basis"])
        np.testing.assert_array_almost_equal(position["vector"], expected["vector"])

    def test_convert_sho_basis_order(self) -> None:
        eigenstate = _get_random_sho_eigenstate((5, 6, 9), (10, 10, 10))
        stacked = _stack_eigenstate(eigenstate)

        expected = _convert_momentum_basis_x01_to_position(
            _convert_explicit_basis_x2_to_position(stacked)
        )
        actual = _convert_explicit_basis_x2_to_position(
            _convert_momentum_basis_x01_to_position(stacked)
        )
        np.testing.assert_array_almost_equal(expected["vector"], actual["vector"])

        actual_flat = _flatten_eigenstate(actual)
        expected_flat = convert_eigenstate_to_basis(eigenstate, actual_flat["basis"])
        np.testing.assert_array_almost_equal(
            expected_flat["vector"], actual_flat["vector"]
        )

    def test_convert_sho_basis_explicit(self) -> None:
        nx = _rng.integers(2, 10)
        fundamental_nx = _rng.integers(nx, nx + 10)
        ny = _rng.integers(2, 10)
        fundamental_ny = _rng.integers(ny, ny + 10)

        nz = _rng.integers(2, 10)
        fundamental_nz = _rng.integers(nz, nz + 10)

        eigenstate = _get_random_sho_eigenstate(
            (nx, ny, nz), (fundamental_nx, fundamental_ny, fundamental_nz)
        )
        stacked = _stack_eigenstate(eigenstate)

        for i in range(nz):
            vector = np.zeros(fundamental_nz)
            vector[i] = 1
            stacked["basis"][2]["vectors"][i] = vector

        position_eigenstate = _convert_explicit_basis_x2_to_position(stacked)

        expected = np.zeros((nx, ny, fundamental_nz), dtype=complex)
        expected[:, :, :nz] = stacked["vector"]

        np.testing.assert_array_equal(expected, position_eigenstate["vector"])

    def test_convert_sho_eigenstate_rust_simple(self) -> None:
        resolution = (5, 6, 9)
        eigenstate = _get_random_sho_eigenstate(resolution, (10, 10, 100))
        config: SHOBasisConfig = {
            "mass": hbar**2,
            "sho_omega": 1 / hbar,
            "x_origin": np.array([0, 0, -10]),
        }

        eigenstate["basis"] = (
            eigenstate["basis"][0],
            eigenstate["basis"][1],
            infinate_sho_basis_from_config(
                eigenstate["basis"][2]["parent"], config, resolution[2]
            ),
        )

        util = BasisConfigUtil(eigenstate["basis"])

        for i in range(resolution[2]):
            vector = np.zeros_like(eigenstate["vector"])
            vector[np.ravel_multi_index((0, 0, i), resolution)] = 1
            eigenstate["vector"] = vector

            points = util.fundamental_x_points + config["x_origin"][:, np.newaxis]

            actual = hamiltonian_generator.get_eigenstate_wavefunction(
                resolution,
                (util.delta_x0.item(0), 0),
                (util.delta_x1.item(0), util.delta_x1.item(1)),
                config["mass"],
                config["sho_omega"],
                0,
                0,
                eigenstate["vector"].tolist(),
                points.T.tolist(),
            )

            expected = _convert_sho_eigenstate_to_position_basis(eigenstate)
            np.testing.assert_allclose(
                expected["vector"], np.array(actual) / np.linalg.norm(actual)
            )

    def test_convert_sho_eigenstate_rust(self) -> None:
        resolution = (5, 6, 9)
        eigenstate = _get_random_sho_eigenstate(resolution, (10, 10, 100))
        config: SHOBasisConfig = {
            "mass": hbar**2,
            "sho_omega": 1 / hbar,
            "x_origin": np.array([0, 0, -10]),
        }

        eigenstate["basis"] = (
            eigenstate["basis"][0],
            eigenstate["basis"][1],
            infinate_sho_basis_from_config(
                eigenstate["basis"][2]["parent"], config, resolution[2]
            ),
        )

        util = BasisConfigUtil(eigenstate["basis"])

        points = util.fundamental_x_points + config["x_origin"][:, np.newaxis]
        actual = hamiltonian_generator.get_eigenstate_wavefunction(
            resolution,
            (util.delta_x0.item(0), 0),
            (util.delta_x1.item(0), util.delta_x1.item(1)),
            config["mass"],
            config["sho_omega"],
            0,
            0,
            eigenstate["vector"].tolist(),
            points.T.tolist(),
        )

        expected = _convert_sho_eigenstate_to_position_basis(eigenstate)
        np.testing.assert_allclose(
            expected["vector"], np.array(actual) / np.linalg.norm(actual)
        )

    def test_convert_truncated_basis_eigenstate(self) -> None:
        resolution = (5, 6, 9)
        eigenstate = _get_random_sho_eigenstate(resolution, (10, 10, 100))

        config: SHOBasisConfig = {
            "mass": hbar**2,
            "sho_omega": 1 / hbar,
            "x_origin": np.array([0, 0, -10]),
        }

        eigenstate["basis"] = (
            eigenstate["basis"][0],
            eigenstate["basis"][1],
            infinate_sho_basis_from_config(
                eigenstate["basis"][2]["parent"], config, resolution[2]
            ),
        )

        actual = _convert_sho_eigenstate_to_position_basis(eigenstate)

        expected = convert_eigenstate_to_basis(eigenstate, actual["basis"])

        np.testing.assert_array_almost_equal(actual["vector"], expected["vector"])

    def test_convert_random_sho_eigenstate(self) -> None:
        resolution = (5, 6, 9)
        eigenstate = _get_random_sho_eigenstate(resolution, (10, 10, 100))

        actual = _convert_sho_eigenstate_to_position_basis(eigenstate)

        expected = convert_eigenstate_to_basis(eigenstate, actual["basis"])
        np.testing.assert_array_almost_equal(actual["vector"], expected["vector"])

    def test_convert_random_explicit_z_eigenstate(self) -> None:
        resolution = (5, 6, 9)
        eigenstate = _get_random_sho_eigenstate(resolution, (10, 10, 100))
        eigenstate["basis"] = (  # type: ignore[typeddict-item]
            as_fundamental_basis(eigenstate["basis"][0]),
            as_fundamental_basis(eigenstate["basis"][1]),
            eigenstate["basis"][2],
        )

        actual = _convert_sho_eigenstate_to_position_basis(eigenstate)

        expected = convert_eigenstate_to_basis(eigenstate, actual["basis"])
        np.testing.assert_array_almost_equal(actual["vector"], expected["vector"])
