from __future__ import annotations

import unittest
from typing import TYPE_CHECKING, Any, TypeVar

import hamiltonian_generator
import numpy as np
import scipy.linalg
import scipy.special
from scipy.constants import hbar

from _tests.utils import convert_explicit_basis_x2
from surface_potential_analysis.basis_config.basis_config import (
    BasisConfigUtil,
    PositionBasisConfig,
    PositionBasisConfigUtil,
)
from surface_potential_analysis.basis_config.sho_basis import (
    SHOBasisConfig,
    calculate_x_distances,
)
from surface_potential_analysis.hamiltonian.hamiltonian import (
    HamiltonianWithBasis,
    flatten_hamiltonian,
    stack_hamiltonian,
)
from surface_potential_analysis.hamiltonian_builder import (
    momentum_basis,
    sho_subtracted_basis,
)
from surface_potential_analysis.hamiltonian_builder.sho_subtracted_basis import (
    _SurfaceHamiltonianUtil,
)
from surface_potential_analysis.interpolation import interpolate_points_rfftn

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import (
        Basis,
        ExplicitBasis,
        MomentumBasis,
        PositionBasis,
    )
    from surface_potential_analysis.potential.potential import Potential

    _BX0Inv = TypeVar("_BX0Inv", bound=Basis[Any, Any])
    _BX1Inv = TypeVar("_BX1Inv", bound=Basis[Any, Any])

    _L0Inv = TypeVar("_L0Inv", bound=int)
    _L1Inv = TypeVar("_L1Inv", bound=int)

rng = np.random.default_rng()


def _convert_x2_to_explicit_basis(
    hamiltonian: HamiltonianWithBasis[_BX0Inv, _BX1Inv, MomentumBasis[_L0Inv]],
    basis: ExplicitBasis[_L1Inv, PositionBasis[_L0Inv]],
) -> HamiltonianWithBasis[
    _BX0Inv, _BX1Inv, ExplicitBasis[_L1Inv, PositionBasis[_L0Inv]]
]:
    stacked = stack_hamiltonian(hamiltonian)

    x2_position = np.fft.fftn(
        np.fft.ifftn(stacked["array"], axes=(2,), norm="ortho"),
        axes=(5,),
        norm="ortho",
    )
    x2_explicit = convert_explicit_basis_x2(x2_position, basis["vectors"])

    return flatten_hamiltonian(
        {
            "basis": (hamiltonian["basis"][0], hamiltonian["basis"][1], basis),
            "array": x2_explicit,  # type: ignore[typeddict-item]
        }
    )


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
) -> np.ndarray[tuple[int, int, int], np.dtype[np.float_]]:
    return np.swapaxes([_generate_random_potential(width) for _ in range(height)], 0, -1)  # type: ignore[no-any-return]


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
    potentail: Potential[Any, Any, Any] = {
        "points": np.zeros((2 * nkx, 2 * nky, nz)),
        "basis": PositionBasisConfigUtil.from_resolution(
            (2 * nkx, 2 * nky, nz),
            (
                np.array([2 * np.pi * hbar, 0, 0]),
                np.array([0, 2 * np.pi * hbar, 0]),
                np.array([0, 0, nz]),
            ),
        ),
    }
    hamiltonian = _SurfaceHamiltonianUtil(potentail, config, resolution)

    potentail2: Potential[Any, Any, Any] = {
        "points": np.tile(hamiltonian.get_sho_potential(), (2 * nkx, 2 * nky, 1)),
        "basis": potentail["basis"],
    }
    return _SurfaceHamiltonianUtil(potentail2, config, resolution)


class HamiltonianBuilderTest(unittest.TestCase):
    def test_diagonal_energies(self) -> None:
        resolution = (2, 2, 2)
        config: SHOBasisConfig = {
            "mass": 1,
            "sho_omega": 1 / hbar,
            "x_origin": np.array([0, 0, -1]),
        }
        potentail: Potential[Any, Any, Any] = {
            "points": np.zeros((4, 4, 3)),
            "basis": PositionBasisConfigUtil.from_resolution(
                (4, 4, 3),
                (
                    np.array([2 * np.pi * hbar, 0, 0]),
                    np.array([0, 2 * np.pi * hbar, 0]),
                    np.array([0, 0, 3]),
                ),
            ),
        }
        hamiltonian = _SurfaceHamiltonianUtil(potentail, config, resolution)

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
        potentail: Potential[Any, Any, Any] = {
            "points": np.zeros((4, 4, 5)),
            "basis": PositionBasisConfigUtil.from_resolution(
                (4, 4, 5),
                (
                    np.array([2 * np.pi * hbar, 0, 0]),
                    np.array([0, 2 * np.pi * hbar, 0]),
                    np.array([0, 0, 5]),
                ),
            ),
        }
        hamiltonian = _SurfaceHamiltonianUtil(potentail, config, resolution)
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
        potentail: Potential[Any, Any, Any] = {
            "points": np.zeros((2 * nx, 2 * ny, nz)),
            "basis": PositionBasisConfigUtil.from_resolution(
                (2 * nx, 2 * ny, nz),
                (
                    np.array([2 * np.pi * hbar, 0, 0]),
                    np.array([0, 2 * np.pi * hbar, 0]),
                    np.array([0, 0, nz]),
                ),
            ),
        }
        hamiltonian = _SurfaceHamiltonianUtil(potentail, config, resolution)

        potentail2: Potential[Any, Any, Any] = {
            "points": np.tile(hamiltonian.get_sho_potential(), (2 * nx, 2 * ny, 1)),
            "basis": PositionBasisConfigUtil.from_resolution(
                (2 * nx, 2 * ny, nz),
                (
                    np.array([2 * np.pi * hbar, 0, 0]),
                    np.array([0, 2 * np.pi * hbar, 0]),
                    np.array([0, 0, nz]),
                ),
            ),
        }

        hamiltonian = _SurfaceHamiltonianUtil(potentail2, config, resolution)
        actual = hamiltonian.get_sho_subtracted_points()
        expected = np.zeros(shape=(2 * nx, 2 * ny, nz))

        np.testing.assert_allclose(expected, actual)

    def test_get_fft_is_real(self) -> None:
        width = rng.integers(1, 10) * 2
        nz = rng.integers(2, 100)

        points = _generate_symmetrical_points(nz, width)
        resolution = (points.shape[0] // 2, points.shape[1] // 2, 2)
        config: SHOBasisConfig = {
            "mass": 1,
            "sho_omega": 1,
            "x_origin": np.array([0, 0, -20]),
        }
        potentail: Potential[Any, Any, Any] = {
            "points": points,
            "basis": PositionBasisConfigUtil.from_resolution(
                (points.shape[0], points.shape[1], nz),
                (
                    np.array([2 * np.pi * hbar, 0, 0]),
                    np.array([0, 2 * np.pi * hbar, 0]),
                    np.array([0, 0, nz]),
                ),
            ),
        }

        hamiltonian = _SurfaceHamiltonianUtil(potentail, config, resolution)
        ft_potential = hamiltonian.get_ft_potential()
        np.testing.assert_almost_equal(
            np.imag(ft_potential), np.zeros_like(ft_potential)
        )
        np.testing.assert_almost_equal(np.real(ft_potential), ft_potential)

    def test_get_fft_normalization(self) -> None:
        hamiltonian = _generate_random_diagonal_hamiltonian()
        z_points = rng.random(hamiltonian.nz)
        hamiltonian._potential["points"][0][0] = [  # noqa: SLF001
            x + o
            for (x, o) in zip(
                hamiltonian._potential["points"][0][0],  # noqa: SLF001
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
        util = BasisConfigUtil(hamiltonian.basis)
        n_points = util.n0 * util.n1 * util.n2
        expected_shape = (n_points, n_points)
        np.testing.assert_equal(actual, np.zeros(shape=expected_shape))

    def test_is_almost_hermitian(self) -> None:
        width = rng.integers(1, 10) * 2
        nz = rng.integers(2, 100)

        points = _generate_symmetrical_points(nz, width)
        np.testing.assert_allclose(points[1:, 1:], points[1:, 1:][::-1, ::-1])
        resolution = (width // 2, width // 2, 10)
        config: SHOBasisConfig = {
            "mass": 1,
            "sho_omega": 1,
            "x_origin": np.array([0, 0, -2]),
        }
        potentail: Potential[Any, Any, Any] = {
            "points": points,
            "basis": PositionBasisConfigUtil.from_resolution(
                (points.shape[0], points.shape[1], nz),
                (
                    np.array([2 * np.pi * hbar, 0, 0]),
                    np.array([0, 2 * np.pi * hbar, 0]),
                    np.array([0, 0, nz]),
                ),
            ),
        }

        hamiltonian = _SurfaceHamiltonianUtil(potentail, config, resolution)

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
        potentail: Potential[Any, Any, Any] = {
            "points": np.zeros(shape=(nx, ny, nz)),
            "basis": PositionBasisConfigUtil.from_resolution(
                (nx, ny, nz),
                (
                    np.array([2 * np.pi * hbar, 0, 0]),
                    np.array([0, 2 * np.pi * hbar, 0]),
                    np.array([0, 0, nz]),
                ),
            ),
        }

        hamiltonian = _SurfaceHamiltonianUtil(potentail, config, resolution)

        np.testing.assert_allclose(
            hamiltonian._calculate_off_diagonal_energies_fast(),  # noqa: SLF001
            hamiltonian._calculate_off_diagonal_energies(),  # noqa: SLF001
        )

    def test_hamiltonian_from_potential(self) -> None:
        shape = rng.integers(1, 10, size=3)
        points = rng.random(shape)

        expected_basis: PositionBasisConfig[Any, Any, Any] = (
            {"n": shape.item(0), "_type": "position", "delta_x": np.array([1.0, 0, 0])},
            {"n": shape.item(1), "_type": "position", "delta_x": np.array([0, 1.0, 0])},
            {"n": shape.item(2), "_type": "position", "delta_x": np.array([0, 0, 1.0])},
        )
        potential: Potential[Any, Any, Any] = {
            "basis": expected_basis,
            "points": points,
        }

        hamiltonian = momentum_basis.hamiltonian_from_potential(potential)

        for ix0 in range(shape[0]):
            for ix1 in range(shape[1]):
                for ix2 in range(shape[2]):
                    np.testing.assert_equal(
                        hamiltonian["array"][ix0, ix1, ix2, ix0, ix1, ix2],
                        potential["points"][ix0, ix1, ix2],
                    )
        np.testing.assert_equal(
            np.count_nonzero(hamiltonian["array"]),
            np.count_nonzero(potential["points"]),
        )

        for expected, actual in zip(expected_basis, hamiltonian["basis"], strict=True):
            assert expected["n"] == actual["n"]
            assert expected["_type"] == actual["_type"]
            np.testing.assert_array_equal(expected["delta_x"], actual["delta_x"])

    def test_total_surface_hamiltonian_simple(self) -> None:
        shape = np.array([3, 3, 200])  # np.random.randint(1, 2, size=3, dtype=int)
        nz = 6

        expected_basis: PositionBasisConfig[Any, Any, Any] = (
            {
                "n": shape.item(0),
                "_type": "position",
                "delta_x": np.array([2 * np.pi, 0, 0]),
            },
            {
                "n": shape.item(1),
                "_type": "position",
                "delta_x": np.array([0, 2 * np.pi, 0]),
            },
            {
                "n": shape.item(2),
                "_type": "position",
                "delta_x": np.array([0, 0, 5 * np.pi]),
            },
        )
        config: SHOBasisConfig = {
            "mass": hbar**2,
            "sho_omega": 1 / hbar,
            "x_origin": np.array([0, 0, -2.5 * np.pi]),
        }
        bloch_phase = np.array([0, 0, 0])

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

        potential: Potential[Any, Any, Any] = {
            "basis": expected_basis,
            "points": points,
        }

        momentum_builder_result = momentum_basis.total_surface_hamiltonian(
            potential, config["mass"], bloch_phase
        )

        interpolated_points = interpolate_points_rfftn(
            points, s=(2 * points.shape[0], 2 * points.shape[1]), axes=(0, 1)
        )

        interpolated_potential: Potential[int, int, int] = {
            "basis": (
                {
                    "n": interpolated_points.shape[0],
                    "_type": "position",
                    "delta_x": potential["basis"][0]["delta_x"],
                },
                {
                    "n": interpolated_points.shape[1],
                    "_type": "position",
                    "delta_x": potential["basis"][1]["delta_x"],
                },
                potential["basis"][2],
            ),
            "points": interpolated_points,  # type: ignore[typeddict-item]
        }

        actual = sho_subtracted_basis.total_surface_hamiltonian(
            interpolated_potential,
            config,
            bloch_phase,
            (points.shape[0], points.shape[1], nz),
        )

        expected = _convert_x2_to_explicit_basis(
            momentum_builder_result, actual["basis"][2]
        )

        np.testing.assert_array_almost_equal(actual["array"], expected["array"])

    def __test_total_surface_hamiltonian(self) -> None:
        shape = np.array([3, 3, 200])
        nz = 6

        expected_basis: PositionBasisConfig[Any, Any, Any] = (
            {
                "n": shape.item(0),
                "_type": "position",
                "delta_x": np.array([2 * np.pi, 0, 0]),
            },
            {
                "n": shape.item(1),
                "_type": "position",
                "delta_x": np.array([0, 2 * np.pi, 0]),
            },
            {
                "n": shape.item(2),
                "_type": "position",
                "delta_x": np.array([0, 0, 5 * np.pi]),
            },
        )
        config: SHOBasisConfig = {
            "mass": hbar**2,
            "sho_omega": 1 / hbar,
            "x_origin": np.array([0, 0, -2.5 * np.pi]),
        }
        bloch_phase = np.array([0, 0, 0])

        points = rng.random(*shape)

        potential: Potential[Any, Any, Any] = {
            "basis": expected_basis,
            "points": points,
        }

        momentum_builder_result = momentum_basis.total_surface_hamiltonian(
            potential, config["mass"], bloch_phase
        )

        interpolated_points = interpolate_points_rfftn(
            points, s=(2 * points.shape[0], 2 * points.shape[1]), axes=(0, 1)
        )

        interpolated_potential: Potential[int, int, int] = {
            "basis": (
                {
                    "n": interpolated_points.shape[0],
                    "_type": "position",
                    "delta_x": potential["basis"][0]["delta_x"],
                },
                {
                    "n": interpolated_points.shape[1],
                    "_type": "position",
                    "delta_x": potential["basis"][1]["delta_x"],
                },
                potential["basis"][2],
            ),
            "points": interpolated_points,  # type: ignore[typeddict-item]
        }

        actual = sho_subtracted_basis.total_surface_hamiltonian(
            interpolated_potential,
            config,
            bloch_phase,
            (points.shape[0], points.shape[1], nz),
        )

        expected = _convert_x2_to_explicit_basis(
            momentum_builder_result, actual["basis"][2]
        )

        np.testing.assert_array_almost_equal(actual["array"], expected["array"])
