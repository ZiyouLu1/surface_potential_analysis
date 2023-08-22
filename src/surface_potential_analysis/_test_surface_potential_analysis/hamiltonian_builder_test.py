from __future__ import annotations

import unittest
from typing import TYPE_CHECKING, Any, Generic, Literal, TypedDict, TypeVar

import hamiltonian_generator
import numpy as np
import scipy.linalg
import scipy.special
from scipy.constants import hbar

from surface_potential_analysis.axis.axis import (
    FundamentalPositionAxis1d,
    FundamentalPositionAxis3d,
    FundamentalTransformedPositionAxis1d,
    FundamentalTransformedPositionAxis3d,
    TransformedPositionAxis,
)
from surface_potential_analysis.basis.basis import (
    Basis3d,
)
from surface_potential_analysis.basis.build import (
    position_basis_3d_from_shape,
)
from surface_potential_analysis.basis.conversion import (
    basis_as_fundamental_momentum_basis,
    basis_as_fundamental_position_basis,
)
from surface_potential_analysis.basis.sho_basis import (
    SHOBasisConfig,
    calculate_x_distances,
)
from surface_potential_analysis.basis.util import (
    AxisWithLengthBasisUtil,
)
from surface_potential_analysis.hamiltonian_builder import (
    momentum_basis,
    sho_subtracted_basis,
)
from surface_potential_analysis.hamiltonian_builder.sho_subtracted_basis import (
    _SurfaceHamiltonianUtil,
)
from surface_potential_analysis.operator.conversion import (
    convert_operator_to_basis,
)
from surface_potential_analysis.potential.conversion import convert_potential_to_basis
from surface_potential_analysis.state_vector.eigenstate_calculation import (
    calculate_eigenvectors_hermitian,
)
from surface_potential_analysis.util.interpolation import interpolate_points_rfftn
from surface_potential_analysis.util.util import slice_along_axis

if TYPE_CHECKING:
    from surface_potential_analysis.axis.axis_like import AxisWithLengthLike3d
    from surface_potential_analysis.basis.basis import (
        FundamentalMomentumBasis3d,
        FundamentalPositionBasis3d,
    )
    from surface_potential_analysis.operator.operator import (
        SingleBasisOperator,
    )
    from surface_potential_analysis.potential.potential import (
        FundamentalPositionBasisPotential3d,
        Potential,
    )

    _L0 = TypeVar("_L0", bound=int)
    _L1 = TypeVar("_L1", bound=int)
    _L2 = TypeVar("_L2", bound=int)

    _L0Inv = TypeVar("_L0Inv", bound=int)
    _L1Inv = TypeVar("_L1Inv", bound=int)
    _L2Inv = TypeVar("_L2Inv", bound=int)

    _LInv = TypeVar("_LInv", bound=int)

    _L0_co = TypeVar("_L0_co", bound=int, covariant=True)
    _L1_co = TypeVar("_L1_co", bound=int, covariant=True)
    _L2_co = TypeVar("_L2_co", bound=int, covariant=True)

    _B3d0Inv = TypeVar("_B3d0Inv", bound=Basis3d[Any, Any, Any])
    _StackedHamiltonianPoints = np.ndarray[
        tuple[_L0_co, _L1_co, _L2_co, _L0_co, _L1_co, _L2_co],
        np.dtype[np.complex_] | np.dtype[np.float_],
    ]
    _A3d0_co = TypeVar("_A3d0_co", bound=AxisWithLengthLike3d[Any, Any], covariant=True)
    _A3d1_co = TypeVar("_A3d1_co", bound=AxisWithLengthLike3d[Any, Any], covariant=True)
    _A3d2_co = TypeVar("_A3d2_co", bound=AxisWithLengthLike3d[Any, Any], covariant=True)

rng = np.random.default_rng()
_B3d0_co = TypeVar("_B3d0_co", bound=Basis3d[Any, Any, Any], covariant=True)


class StackedHamiltonian3d(TypedDict, Generic[_B3d0_co]):
    """Represents an operator with it's array of points 'stacked'."""

    basis: _B3d0_co
    # We need higher kinded types to do this properly
    array: _StackedHamiltonianPoints[int, int, int]


if TYPE_CHECKING:
    StackedHamiltonianWith3dBasis = StackedHamiltonian3d[
        Basis3d[_A3d0_co, _A3d1_co, _A3d2_co]
    ]
    FundamentalMomentumBasisStackedHamiltonian3d = StackedHamiltonian3d[
        FundamentalMomentumBasis3d[_L0_co, _L1_co, _L2_co]
    ]
    FundamentalPositionBasisStackedHamiltonian3d = StackedHamiltonian3d[
        FundamentalPositionBasis3d[_L0_co, _L1_co, _L2_co]
    ]


def convert_explicit_basis_x2(
    hamiltonian: _StackedHamiltonianPoints[_L0Inv, _L1Inv, _L2Inv],
    basis: np.ndarray[tuple[_LInv, _L2Inv], np.dtype[np.complex_]],
) -> _StackedHamiltonianPoints[_L0Inv, _L1Inv, _LInv]:
    end_dot = np.sum(
        hamiltonian[slice_along_axis(np.newaxis, -2)]
        * basis.reshape(1, 1, 1, 1, 1, *basis.shape),
        axis=-1,
    )
    return np.sum(  # type: ignore[no-any-return]
        end_dot[slice_along_axis(np.newaxis, 2)]
        * basis.conj().reshape(1, 1, *basis.shape, 1, 1, 1),
        axis=3,
    )


def flatten_hamiltonian(
    hamiltonian: StackedHamiltonian3d[_B3d0Inv],
) -> SingleBasisOperator[_B3d0Inv]:
    """
    Convert a stacked hamiltonian to a hamiltonian.

    Parameters
    ----------
    hamiltonian : StackedHamiltonian[_B3d0Inv]

    Returns
    -------
    Hamiltonian[_B3d0Inv]
    """
    n_states = np.prod(hamiltonian["array"].shape[:3])
    return {
        "basis": hamiltonian["basis"],
        "dual_basis": hamiltonian["basis"],
        "array": hamiltonian["array"].reshape(n_states, n_states),
    }


def stack_hamiltonian(
    hamiltonian: SingleBasisOperator[_B3d0Inv],
) -> StackedHamiltonian3d[_B3d0Inv]:
    """
    Convert a hamiltonian to a stacked hamiltonian.

    Parameters
    ----------
    hamiltonian : Hamiltonian[_B3d0Inv]

    Returns
    -------
    StackedHamiltonian[_B3d0Inv]
    """
    basis = AxisWithLengthBasisUtil(hamiltonian["basis"])
    return {
        "basis": hamiltonian["basis"],
        "array": hamiltonian["array"].reshape(*basis.shape, *basis.shape),
    }


def hamiltonian_from_position_basis_potential_3d_stacked(
    potential: FundamentalPositionBasisPotential3d[_L0, _L1, _L2],
) -> FundamentalPositionBasisStackedHamiltonian3d[_L0, _L1, _L2]:
    """Given a potential in position basis [ix0, ix1, ix2], get the hamiltonian in stacked form.

    This is just a matrix with the potential along the diagonals

    Parameters
    ----------
    potential : NDArray
        The potential in position basis [ix0, ix1, ix2]

    Returns
    -------
    HamiltonianStacked[PositionBasis, PositionBasis, PositionBasis]
        The hamiltonian in stacked form
    """
    shape = AxisWithLengthBasisUtil(potential["basis"]).shape
    array = np.diag(potential["vector"]).reshape(*shape, *shape)
    return {"basis": potential["basis"], "array": array}


def _generate_random_potential(
    width: int = 5,
) -> np.ndarray[tuple[int, int], np.dtype[np.float_]]:
    random_array = rng.random((width + 1, width + 1))

    out = np.zeros_like(random_array, dtype=float)
    out += random_array[::+1, ::+1]
    out += random_array[::-1, ::+1]
    out += random_array[::+1, ::-1]
    out += random_array[::-1, ::-1]
    out += random_array[::+1, ::+1].T
    out += random_array[::-1, ::+1].T
    out += random_array[::+1, ::-1].T
    out += random_array[::-1, ::-1].T
    return out[:width, :width]  # type: ignore[no-any-return]


def _generate_symmetrical_points(
    height: int, width: int = 5
) -> np.ndarray[tuple[int], np.dtype[np.complex_]]:
    return np.swapaxes([_generate_random_potential(width) for _ in range(height)], 0, -1).ravel()  # type: ignore[no-any-return]


def _generate_random_diagonal_hamiltonian() -> (
    _SurfaceHamiltonianUtil[Any, Any, Any, Any, Any, Any]
):
    nkx = rng.integers(3, 5)
    nky = rng.integers(3, 5)
    nkz = rng.integers(3, 5)

    nz = rng.integers(5, 100)
    z_offset = 20 * rng.random()
    resolution = (nkx, nky, nkz)
    config: SHOBasisConfig = {
        "mass": 1,
        "sho_omega": 1,
        "x_origin": np.array([0, 0, -z_offset]),
    }
    potential: FundamentalPositionBasisPotential3d[Any, Any, Any] = {
        "vector": np.zeros((2 * nkx, 2 * nky, nz)).reshape(-1),
        "basis": position_basis_3d_from_shape(
            (2 * nkx, 2 * nky, nz),
            np.array(
                [
                    np.array([2 * np.pi * hbar, 0, 0]),
                    np.array([0, 2 * np.pi * hbar, 0]),
                    np.array([0, 0, nz]),
                ]
            ),
        ),
    }
    hamiltonian = _SurfaceHamiltonianUtil(potential, config, resolution)

    potential2: FundamentalPositionBasisPotential3d[Any, Any, Any] = {
        "vector": np.tile(
            hamiltonian.get_sho_potential(), (2 * nkx, 2 * nky, 1)
        ).ravel(),
        "basis": potential["basis"],
    }
    return _SurfaceHamiltonianUtil(potential2, config, resolution)


class HamiltonianBuilderTest(unittest.TestCase):
    def test_flatten_hamiltonian(self) -> None:
        shape = rng.integers(1, 10, size=3)
        hamiltonian: FundamentalMomentumBasisStackedHamiltonian3d[int, int, int] = {
            "array": rng.random((*shape, *shape)),
            "basis": (
                FundamentalTransformedPositionAxis3d(
                    np.array([1.0, 0, 0]), shape.item(0)
                ),
                FundamentalTransformedPositionAxis3d(
                    np.array([0, 1.0, 0]), shape.item(1)
                ),
                FundamentalTransformedPositionAxis3d(
                    np.array([0, 0, 1.0]), shape.item(2)
                ),
            ),
        }
        actual = flatten_hamiltonian(hamiltonian)
        expected = np.zeros((np.prod(shape), np.prod(shape)))
        x0t, x1t, zt = np.meshgrid(
            np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
        )
        coords = np.array([x0t.ravel(), x1t.ravel(), zt.ravel()]).T
        for i, (ix0, ix1, ix2) in enumerate(coords):
            for j, (jx0, jx1, jx2) in enumerate(coords):
                expected[i, j] = hamiltonian["array"][ix0, ix1, ix2, jx0, jx1, jx2]

        np.testing.assert_array_equal(actual["array"], expected)

    def test_stack_hamiltonian(self) -> None:
        shape = rng.integers(1, 10, size=3)
        hamiltonian: FundamentalMomentumBasisStackedHamiltonian3d[int, int, int] = {
            "array": rng.random((*shape, *shape)),
            "basis": (
                FundamentalTransformedPositionAxis3d(
                    np.array([1.0, 0, 0]), shape.item(0)
                ),
                FundamentalTransformedPositionAxis3d(
                    np.array([0, 1.0, 0]), shape.item(1)
                ),
                FundamentalTransformedPositionAxis3d(
                    np.array([0, 0, 1.0]), shape.item(2)
                ),
            ),
        }

        actual = stack_hamiltonian(flatten_hamiltonian(hamiltonian))
        np.testing.assert_array_equal(hamiltonian["array"], actual["array"])

    def test_hamiltonian_from_potential_momentum(self) -> None:
        potential: Potential[
            tuple[FundamentalTransformedPositionAxis1d[Literal[100]]]
        ] = {
            "basis": (FundamentalTransformedPositionAxis1d(np.array([1]), 100),),
            "vector": np.array(rng.random(100), dtype=complex),
        }
        actual = momentum_basis.hamiltonian_from_potential(potential)

        converted = convert_potential_to_basis(
            potential, basis_as_fundamental_position_basis(potential["basis"])
        )
        expected = convert_operator_to_basis(
            {
                "basis": converted["basis"],
                "dual_basis": converted["basis"],
                "array": np.diag(converted["vector"]),
            },
            potential["basis"],
            potential["basis"],
        )
        np.testing.assert_array_almost_equal(expected["array"], actual["array"])

    def test_diagonal_energies(self) -> None:
        resolution = (2, 2, 2)
        config: SHOBasisConfig = {
            "mass": 1,
            "sho_omega": 1 / hbar,
            "x_origin": np.array([0, 0, -1]),
        }
        potential: FundamentalPositionBasisPotential3d[Any, Any, Any] = {
            "vector": np.zeros(4 * 4 * 3),
            "basis": position_basis_3d_from_shape(
                (4, 4, 3),
                np.array(
                    [
                        np.array([2 * np.pi * hbar, 0, 0]),
                        np.array([0, 2 * np.pi * hbar, 0]),
                        np.array([0, 0, 3]),
                    ]
                ),
            ),
        }
        hamiltonian = _SurfaceHamiltonianUtil(potential, config, resolution)

        expected = np.array([0.5, 1.5, 1.0, 2.0, 1.0, 2.0, 1.5, 2.5])
        diagonal_energy = hamiltonian._calculate_diagonal_energy(  # noqa: SLF001
            np.array([0, 0, 0])
        )

        np.testing.assert_array_almost_equal(diagonal_energy, expected)

    def test_get_sho_potential(self) -> None:
        resolution = (2, 2, 2)
        config: SHOBasisConfig = {
            "mass": 1,
            "sho_omega": 1,
            "x_origin": np.array([0, 0, -2]),
        }
        potential: FundamentalPositionBasisPotential3d[Any, Any, Any] = {
            "vector": np.zeros((4, 4, 5)).ravel(),
            "basis": position_basis_3d_from_shape(
                (4, 4, 5),
                np.array(
                    [
                        np.array([2 * np.pi * hbar, 0, 0]),
                        np.array([0, 2 * np.pi * hbar, 0]),
                        np.array([0, 0, 5]),
                    ]
                ),
            ),
        }
        hamiltonian = _SurfaceHamiltonianUtil(potential, config, resolution)
        expected = [2.0, 0.5, 0.0, 0.5, 2.0]
        np.testing.assert_equal(expected, hamiltonian.get_sho_potential())

    def test_get_sho_subtracted_points(self) -> None:
        nx = rng.integers(2, 20)
        ny = rng.integers(2, 20)
        nz = rng.integers(2, 100)

        resolution = (nx, ny, 2)
        config: SHOBasisConfig = {
            "mass": 1,
            "sho_omega": 1,
            "x_origin": np.array([0, 0, -20]),
        }
        potential: FundamentalPositionBasisPotential3d[Any, Any, Any] = {
            "vector": np.zeros((2 * nx, 2 * ny, nz)).ravel(),
            "basis": position_basis_3d_from_shape(
                (2 * nx, 2 * ny, nz),
                np.array(
                    [
                        np.array([2 * np.pi * hbar, 0, 0]),
                        np.array([0, 2 * np.pi * hbar, 0]),
                        np.array([0, 0, nz]),
                    ]
                ),
            ),
        }
        hamiltonian = _SurfaceHamiltonianUtil(potential, config, resolution)

        potential2: FundamentalPositionBasisPotential3d[Any, Any, Any] = {
            "vector": np.tile(
                hamiltonian.get_sho_potential(), (2 * nx, 2 * ny, 1)
            ).ravel(),
            "basis": position_basis_3d_from_shape(
                (2 * nx, 2 * ny, nz),
                np.array(
                    [
                        np.array([2 * np.pi * hbar, 0, 0]),
                        np.array([0, 2 * np.pi * hbar, 0]),
                        np.array([0, 0, nz]),
                    ]
                ),
            ),
        }

        hamiltonian = _SurfaceHamiltonianUtil(potential2, config, resolution)
        actual = hamiltonian.get_sho_subtracted_points()
        expected = np.zeros(shape=(2 * nx, 2 * ny, nz))

        np.testing.assert_allclose(expected, actual)

    def test_get_fft_is_real(self) -> None:
        width = rng.integers(1, 10) * 2
        nz = rng.integers(2, 100)

        points = _generate_symmetrical_points(nz, width)
        resolution = (width // 2, width // 2, 2)
        config: SHOBasisConfig = {
            "mass": 1,
            "sho_omega": 1,
            "x_origin": np.array([0, 0, -20]),
        }
        potential: FundamentalPositionBasisPotential3d[Any, Any, Any] = {
            "vector": points,
            "basis": position_basis_3d_from_shape(
                (width, width, nz),
                np.array(
                    [
                        np.array([2 * np.pi * hbar, 0, 0]),
                        np.array([0, 2 * np.pi * hbar, 0]),
                        np.array([0, 0, nz]),
                    ]
                ),
            ),
        }

        hamiltonian = _SurfaceHamiltonianUtil(potential, config, resolution)
        ft_potential = hamiltonian.get_ft_potential()
        np.testing.assert_almost_equal(
            np.imag(ft_potential), np.zeros_like(ft_potential)
        )
        np.testing.assert_almost_equal(np.real(ft_potential), ft_potential)

    def test_get_fft_normalization(self) -> None:
        hamiltonian = _generate_random_diagonal_hamiltonian()
        z_points = rng.random(hamiltonian.nz)
        hamiltonian.points[0][0] = [
            x + o
            for (x, o) in zip(
                hamiltonian.points[0][0],
                z_points,
                strict=True,
            )
        ]

        # fft should pick up a 1/v factor
        ft_potential = hamiltonian.get_ft_potential()
        for iz in range(hamiltonian.nz):
            self.assertAlmostEqual(np.sum(ft_potential[:, :, iz]), z_points[iz])
            ft_value = z_points[iz] / (hamiltonian.nx * hamiltonian.ny)
            np.testing.assert_allclose(ft_potential[:, :, iz], ft_value)

    def test_get_off_diagonal_energies_zero(self) -> None:
        hamiltonian = _generate_random_diagonal_hamiltonian()

        actual = hamiltonian._calculate_off_diagonal_energies()  # noqa: SLF001
        util = AxisWithLengthBasisUtil(hamiltonian.basis)
        n_points = util.shape[0] * util.shape[1] * util.shape[2]  # type: ignore[misc]
        expected_shape = (n_points, n_points)
        np.testing.assert_equal(actual, np.zeros(shape=expected_shape))

    def test_is_almost_hermitian(self) -> None:
        width = rng.integers(1, 10) * 2
        nz = rng.integers(2, 100)

        points = _generate_symmetrical_points(nz, width)
        np.testing.assert_allclose(
            points.reshape(width, width, nz)[1:, 1:],
            points.reshape(width, width, nz)[1:, 1:][::-1, ::-1],
        )
        resolution = (width // 2, width // 2, 10)
        config: SHOBasisConfig = {
            "mass": 1,
            "sho_omega": 1,
            "x_origin": np.array([0, 0, -2]),
        }
        potential: FundamentalPositionBasisPotential3d[Any, Any, Any] = {
            "vector": points,
            "basis": position_basis_3d_from_shape(
                (width, width, nz),
                np.array(
                    [
                        np.array([2 * np.pi * hbar, 0, 0]),
                        np.array([0, 2 * np.pi * hbar, 0]),
                        np.array([0, 0, nz]),
                    ]
                ),
            ),
        }

        hamiltonian = _SurfaceHamiltonianUtil(potential, config, resolution)

        np.testing.assert_allclose(
            hamiltonian.hamiltonian(np.array([0, 0, 0]))["array"],
            hamiltonian.hamiltonian(np.array([0, 0, 0]))["array"].conjugate().T,
        )

    def test_get_hermite_val_rust(self) -> None:
        n = rng.integers(1, 10)
        x = (rng.random() * 10) - 5
        self.assertAlmostEqual(
            hamiltonian_generator.get_hermite_val(x, n),
            scipy.special.eval_hermite(n, [x]).item(0),
            places=6,
        )

    def test_calculate_off_diagonal_energies_rust(self) -> None:
        nx = rng.integers(2, 20)
        ny = rng.integers(2, 20)
        nz = 100

        resolution = (nx // 2, ny // 2, 14)
        config: SHOBasisConfig = {
            "mass": hbar**2,
            "sho_omega": 1 / hbar,
            "x_origin": np.array([0, 0, 0]),
        }
        potential: FundamentalPositionBasisPotential3d[Any, Any, Any] = {
            "vector": np.zeros(shape=(nx, ny, nz)).ravel(),
            "basis": position_basis_3d_from_shape(
                (nx, ny, nz),
                np.array(
                    [
                        np.array([2 * np.pi * hbar, 0, 0]),
                        np.array([0, 2 * np.pi * hbar, 0]),
                        np.array([0, 0, nz]),
                    ]
                ),
            ),
        }

        hamiltonian = _SurfaceHamiltonianUtil(potential, config, resolution)

        np.testing.assert_allclose(
            hamiltonian._calculate_off_diagonal_energies_fast(),  # noqa: SLF001
            hamiltonian._calculate_off_diagonal_energies(),  # noqa: SLF001
        )

    def test_hamiltonian_from_potential(self) -> None:
        shape = rng.integers(1, 10, size=3)
        points = rng.random(shape).ravel()

        expected_basis = position_basis_3d_from_shape(tuple(shape))  # type: ignore[var-annotated,arg-type]

        potential: FundamentalPositionBasisPotential3d[Any, Any, Any] = {
            "basis": expected_basis,
            "vector": np.array(points, dtype=complex),
        }

        hamiltonian = hamiltonian_from_position_basis_potential_3d_stacked(potential)

        for ix0 in range(shape[0]):
            for ix1 in range(shape[1]):
                for ix2 in range(shape[2]):
                    np.testing.assert_equal(
                        hamiltonian["array"][ix0, ix1, ix2, ix0, ix1, ix2],
                        potential["vector"].reshape(shape)[ix0, ix1, ix2],
                    )
        np.testing.assert_equal(
            np.count_nonzero(hamiltonian["array"]),
            np.count_nonzero(potential["vector"]),
        )

        for expected, actual in zip(expected_basis, hamiltonian["basis"], strict=True):
            assert expected.n == actual.n
            assert isinstance(actual, type(expected))
            np.testing.assert_array_equal(expected.delta_x, actual.delta_x)

    def test_total_surface_hamiltonian_simple(self) -> None:
        shape = np.array([3, 3, 200])  # np.random.randint(1, 2, size=3, dtype=int)
        nz = 6

        expected_basis: FundamentalPositionBasis3d[Any, Any, Any] = (
            FundamentalPositionAxis3d(np.array([2 * np.pi, 0, 0]), shape.item(0)),
            FundamentalPositionAxis3d(np.array([0, 2 * np.pi, 0]), shape.item(1)),
            FundamentalPositionAxis3d(np.array([0, 0, 5 * np.pi]), shape.item(2)),
        )
        config: SHOBasisConfig = {
            "mass": hbar**2,
            "sho_omega": 1 / hbar,
            "x_origin": np.array([0, 0, -2.5 * np.pi]),
        }
        bloch_fraction = np.array([0, 0, 0])

        points = np.tile(
            (
                0.5
                * config["mass"]
                * config["sho_omega"] ** 2
                * np.square(
                    calculate_x_distances(expected_basis[2], config["x_origin"])
                )
            ),
            (shape.item(0), shape.item(1), 1),
        )

        potential: FundamentalPositionBasisPotential3d[Any, Any, int] = {
            "basis": expected_basis,
            "vector": points.ravel(),
        }

        momentum_builder_result = momentum_basis.total_surface_hamiltonian(
            potential, config["mass"], bloch_fraction
        )

        interpolated_points = interpolate_points_rfftn(
            points, s=(2 * points.shape[0], 2 * points.shape[1]), axes=(0, 1)
        )

        interpolated_potential: FundamentalPositionBasisPotential3d[int, int, int] = {
            "basis": (
                FundamentalPositionAxis3d(
                    potential["basis"][0].delta_x, interpolated_points.shape[0]
                ),
                FundamentalPositionAxis3d(
                    potential["basis"][1].delta_x, interpolated_points.shape[1]
                ),
                FundamentalPositionAxis3d(
                    potential["basis"][2].delta_x, interpolated_points.shape[2]
                ),
            ),
            "vector": interpolated_points,  # type: ignore[typeddict-item]
        }

        actual = sho_subtracted_basis.total_surface_hamiltonian(
            interpolated_potential,
            config,
            bloch_fraction,
            (points.shape[0], points.shape[1], nz),
        )

        expected = convert_operator_to_basis(
            momentum_builder_result, actual["basis"], actual["dual_basis"]
        )

        np.testing.assert_array_almost_equal(actual["array"], expected["array"])

    def __test_total_surface_hamiltonian(self) -> None:
        shape = np.array([3, 3, 200])
        nz = 6

        expected_basis: FundamentalPositionBasis3d[Any, Any, Any] = (
            FundamentalPositionAxis3d(np.array([2 * np.pi, 0, 0]), shape.item(0)),
            FundamentalPositionAxis3d(np.array([0, 2 * np.pi, 0]), shape.item(1)),
            FundamentalPositionAxis3d(np.array([0, 0, 5 * np.pi]), shape.item(2)),
        )
        config: SHOBasisConfig = {
            "mass": hbar**2,
            "sho_omega": 1 / hbar,
            "x_origin": np.array([0, 0, -2.5 * np.pi]),
        }
        bloch_fraction = np.array([0, 0, 0])

        points = rng.random(*shape)

        potential: FundamentalPositionBasisPotential3d[Any, Any, Any] = {
            "basis": expected_basis,
            "vector": points.ravel(),
        }

        momentum_builder_result = momentum_basis.total_surface_hamiltonian(
            potential, config["mass"], bloch_fraction
        )

        interpolated_points = interpolate_points_rfftn(
            points, s=(2 * points.shape[0], 2 * points.shape[1]), axes=(0, 1)
        )

        interpolated_potential: FundamentalPositionBasisPotential3d[int, int, int] = {
            "basis": (
                FundamentalPositionAxis3d(
                    potential["basis"][0].delta_x, interpolated_points.shape[0]
                ),
                FundamentalPositionAxis3d(
                    potential["basis"][1].delta_x, interpolated_points.shape[1]
                ),
                FundamentalPositionAxis3d(
                    potential["basis"][2].delta_x, interpolated_points.shape[2]
                ),
            ),
            "points": interpolated_points,  # type: ignore[typeddict-item]
        }

        actual = sho_subtracted_basis.total_surface_hamiltonian(
            interpolated_potential,
            config,
            bloch_fraction,
            (points.shape[0], points.shape[1], nz),
        )

        expected = convert_operator_to_basis(
            momentum_builder_result, actual["basis"], actual["dual_basis"]
        )

        np.testing.assert_array_almost_equal(actual["array"], expected["array"])

    def test_momentum_builder_sho_hamiltonian(self) -> None:
        mass = hbar**2
        omega = 1 / hbar
        basis = (FundamentalPositionAxis1d(np.array([30]), 1000),)
        util = AxisWithLengthBasisUtil(basis)
        potential: Potential[tuple[FundamentalPositionAxis1d[int]]] = {
            "basis": basis,
            "vector": 0.5
            * mass
            * omega**2
            * np.linalg.norm(util.x_points - 15, axis=0) ** 2,
        }
        hamiltonian = momentum_basis.total_surface_hamiltonian(
            potential, mass, np.array([0])
        )
        eigenstates = calculate_eigenvectors_hermitian(
            hamiltonian, subset_by_index=(0, 50)
        )
        expected = hbar * omega * (util.nx_points[0] + 0.5)
        np.testing.assert_almost_equal(expected[:50], eigenstates["eigenvalues"][:50])

        in_basis = convert_potential_to_basis(
            potential, basis_as_fundamental_momentum_basis(potential["basis"])
        )
        hamiltonian2 = momentum_basis.total_surface_hamiltonian(
            in_basis, mass, np.array([0])
        )
        eigenstates2 = calculate_eigenvectors_hermitian(
            hamiltonian2, subset_by_index=(0, 50)
        )
        np.testing.assert_almost_equal(expected[:50], eigenstates2["eigenvalues"][:50])

        extended: Potential[tuple[TransformedPositionAxis[int, int, int]]] = {
            "basis": (TransformedPositionAxis(np.array([30]), 1000, 2000),),
            "vector": in_basis["vector"] * np.sqrt(2000 / 1000),
        }
        converted = convert_potential_to_basis(
            extended, basis_as_fundamental_momentum_basis(extended["basis"])
        )
        hamiltonian3 = momentum_basis.total_surface_hamiltonian(
            converted, mass, np.array([0])
        )
        eigenstates3 = calculate_eigenvectors_hermitian(
            hamiltonian3, subset_by_index=(0, 50)
        )
        np.testing.assert_almost_equal(expected[:50], eigenstates3["eigenvalues"][:50])
