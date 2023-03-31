import random
import unittest
from typing import Any

import hamiltonian_generator
import numpy as np
import scipy.linalg
import scipy.special
from scipy.constants import hbar

from surface_potential_analysis.basis_config import (
    BasisConfigUtil,
    PositionBasisConfigUtil,
)
from surface_potential_analysis.hamiltonian_builder.sho_basis import (
    SurfaceHamiltonianUtil,
)
from surface_potential_analysis.potential.potential import Potential
from surface_potential_analysis.sho_basis import SHOBasisConfig


def generate_random_potential(
    width: int = 5,
) -> np.ndarray[tuple[int, int], np.dtype[np.float_]]:
    random_array = np.random.rand(width + 1, width + 1)

    out = np.zeros_like(random_array, dtype=float)
    out += random_array[::+1, ::+1]
    out += random_array[::-1, ::+1]
    out += random_array[::+1, ::-1]
    out += random_array[::-1, ::-1]
    out += random_array[::+1, ::+1].T
    out += random_array[::-1, ::+1].T
    out += random_array[::+1, ::-1].T
    out += random_array[::-1, ::-1].T
    return out[:width, :width]  # type:ignore


def generate_symmetrical_points(
    height: int, width: int = 5
) -> np.ndarray[tuple[int, int, int], np.dtype[np.float_]]:
    return np.swapaxes([generate_random_potential(width) for _ in range(height)], 0, -1)  # type: ignore


def generate_random_diagonal_hamiltonian() -> (
    SurfaceHamiltonianUtil[Any, Any, Any, Any, Any, Any]
):
    nkx = random.randrange(3, 10)
    nky = random.randrange(3, 10)
    nkz = random.randrange(3, 10)

    nz = random.randrange(5, 100)
    z_offset = 20 * random.random()
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
    hamiltonian = SurfaceHamiltonianUtil(potentail, config, resolution)

    potentail2: Potential[Any, Any, Any] = {
        "points": np.tile(hamiltonian.get_sho_potential(), (2 * nkx, 2 * nky, 1)),
        "basis": potentail["basis"],
    }
    return SurfaceHamiltonianUtil(potentail2, config, resolution)


class TestSHOHamiltonian(unittest.TestCase):
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
        hamiltonian = SurfaceHamiltonianUtil(potentail, config, resolution)

        expected = np.array([0.5, 1.5, 1.0, 2.0, 1.0, 2.0, 1.5, 2.5])
        diagonal_energy = hamiltonian._calculate_diagonal_energy(0, 0)

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
                (4, 4, 3),
                (
                    np.array([2 * np.pi * hbar, 0, 0]),
                    np.array([0, 2 * np.pi * hbar, 0]),
                    np.array([0, 0, 3]),
                ),
            ),
        }
        hamiltonian = SurfaceHamiltonianUtil(potentail, config, resolution)
        expected = [2.0, 0.5, 0.0, 0.5, 2.0]
        np.testing.assert_equal(expected, hamiltonian.get_sho_potential())

    def test_get_sho_subtracted_points(self) -> None:
        nx = random.randrange(2, 20)
        ny = random.randrange(2, 20)
        nz = random.randrange(2, 100)

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
        hamiltonian = SurfaceHamiltonianUtil(potentail, config, resolution)

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

        hamiltonian = SurfaceHamiltonianUtil(potentail2, config, resolution)
        actual = hamiltonian.get_sho_subtracted_points()
        expected = np.zeros(shape=(2 * nx, 2 * ny, nz))

        np.testing.assert_allclose(expected, actual)

    def test_get_fft_is_real(self) -> None:
        width = random.randrange(1, 10) * 2
        nz = random.randrange(2, 100)

        points = generate_symmetrical_points(nz, width)
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

        hamiltonian = SurfaceHamiltonianUtil(potentail, config, resolution)
        ft_potential = hamiltonian.get_ft_potential()
        np.testing.assert_almost_equal(
            np.imag(ft_potential), np.zeros_like(ft_potential)
        )
        np.testing.assert_almost_equal(np.real(ft_potential), ft_potential)

    def test_get_fft_normalization(self) -> None:
        hamiltonian = generate_random_diagonal_hamiltonian()
        z_points = np.random.rand(hamiltonian.Nz)
        hamiltonian._potential["points"][0][0] = [
            x + o for (x, o) in zip(hamiltonian._potential["points"][0][0], z_points)
        ]

        # fft should pick up a 1/v factor
        ft_potential = hamiltonian.get_ft_potential()
        for iz in range(hamiltonian.Nz):
            self.assertAlmostEqual(np.sum(ft_potential[:, :, iz]), z_points[iz])
            ft_value = z_points[iz] / (hamiltonian.Nx * hamiltonian.Ny)
            np.testing.assert_allclose(ft_potential[:, :, iz], ft_value)

    def test_get_off_diagonal_energies_zero(self) -> None:
        hamiltonian = generate_random_diagonal_hamiltonian()

        actual = hamiltonian._calculate_off_diagonal_energies()
        util = BasisConfigUtil(hamiltonian.basis)
        n_points = util.n0 * util.n1 * util.n2
        expected_shape = (n_points, n_points)
        np.testing.assert_equal(actual, np.zeros(shape=expected_shape))

    def test_is_almost_hermitian(self) -> None:
        width = random.randrange(1, 10) * 2
        nz = random.randrange(2, 100)

        points = generate_symmetrical_points(nz, width)
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

        hamiltonian = SurfaceHamiltonianUtil(potentail, config, resolution)

        np.testing.assert_allclose(
            hamiltonian.hamiltonian(0, 0)["array"],
            hamiltonian.hamiltonian(0, 0)["array"].conjugate().T,
        )

    def test_sho_normalization(self) -> None:
        nx = random.randrange(2, 10)
        ny = random.randrange(2, 10)
        nz = 1001
        z_width = 20

        z_offset = -z_width / 2

        resolution = (nx, ny, 10)
        config: SHOBasisConfig = {
            "mass": hbar**2,
            "sho_omega": 1 / hbar,
            "x_origin": np.array([0, 0, z_offset]),
        }
        potentail: Potential[Any, Any, Any] = {
            "points": np.zeros(shape=(2 * nx, 2 * ny, nz)),
            "basis": PositionBasisConfigUtil.from_resolution(
                (2 * nx, 2 * ny, nz),
                (
                    np.array([2 * np.pi * hbar, 0, 0]),
                    np.array([0, 2 * np.pi * hbar, 0]),
                    np.array([0, 0, z_width]),
                ),
            ),
        }

        hamiltonian = SurfaceHamiltonianUtil(potentail, config, resolution)

        for iz1 in range(12):
            for iz2 in range(12):
                sho_1 = hamiltonian._calculate_sho_wavefunction_points(iz1)
                sho_2 = hamiltonian._calculate_sho_wavefunction_points(iz2)
                sho_norm = hamiltonian.dz * np.sum(sho_1 * sho_2, dtype=float)

                if iz1 == iz2:
                    self.assertAlmostEqual(sho_norm, 1.0)
                else:
                    self.assertAlmostEqual(sho_norm, 0.0)

    def test_get_hermite_val_rust(self) -> None:
        n = random.randrange(1, 10)
        x = random.random() * 10 - 5
        self.assertAlmostEqual(
            hamiltonian_generator.get_hermite_val(x, n),
            scipy.special.eval_hermite(n, x),
            places=6,
        )

    def test_calculate_off_diagonal_energies_rust(self) -> None:
        nx = random.randrange(2, 20)
        ny = random.randrange(2, 20)
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

        hamiltonian = SurfaceHamiltonianUtil(potentail, config, resolution)

        np.testing.assert_allclose(
            hamiltonian._calculate_off_diagonal_energies_fast(),
            hamiltonian._calculate_off_diagonal_energies(),
        )


if __name__ == "__main__":
    unittest.main()
