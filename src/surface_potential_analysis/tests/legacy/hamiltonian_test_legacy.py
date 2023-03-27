import random
import unittest

import numpy as np
import scipy.linalg
import scipy.special
from scipy.constants import hbar

import hamiltonian_generator
from surface_potential_analysis._legacy.energy_data import EnergyInterpolation
from surface_potential_analysis._legacy.energy_eigenstate import EigenstateConfig
from surface_potential_analysis._legacy.hamiltonian import (
    SurfaceHamiltonianUtil,
    calculate_sho_wavefunction,
    flatten_hamiltonian,
    get_hamiltonian_from_potential,
    unflatten_hamiltonian,
)


def generate_random_potential(width=5):
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
    return out[:width, :width]


def generate_symmetrical_points(height, width=5):
    return np.swapaxes([generate_random_potential(width) for _ in range(height)], 0, -1)


def generate_random_diagonal_hamiltonian() -> SurfaceHamiltonianUtil:
    nkx = random.randrange(3, 10)
    nky = random.randrange(3, 10)
    nkz = random.randrange(3, 10)

    nz = random.randrange(5, 100)
    z_offset = 20 * random.random()
    config: EigenstateConfig = {
        "mass": 1,
        "sho_omega": 1,
        "delta_x0": (2 * np.pi * hbar, 0),
        "delta_x1": (0, 2 * np.pi * hbar),
        "resolution": (nkx, nky, nkz),
    }
    data: EnergyInterpolation = {
        "points": np.zeros(shape=(2 * nkx, 2 * nky, nz)).tolist(),
        "dz": 1,
    }
    hamiltonian = SurfaceHamiltonianUtil(config, data, z_offset)

    data2: EnergyInterpolation = {
        "points": np.tile(
            hamiltonian.get_sho_potential(), (2 * nkx, 2 * nky, 1)
        ).tolist(),
        "dz": 1,
    }
    return SurfaceHamiltonianUtil(config, data2, z_offset)


class TestSurfaceHamiltonian(unittest.TestCase):
    def test_diagonal_energies(self) -> None:
        z_offset = 0
        config: EigenstateConfig = {
            "mass": 1,
            "sho_omega": 1 / hbar,
            "delta_x0": (2 * np.pi * hbar, 0),
            "delta_x1": (0, 2 * np.pi * hbar),
            "resolution": (2, 2, 2),
        }
        data: EnergyInterpolation = {
            "points": np.zeros((4, 4, 3)).tolist(),
            "dz": 1,
        }
        hamiltonian = SurfaceHamiltonianUtil(config, data, z_offset)

        expected = np.array([0.5, 1.5, 1.0, 2.0, 1.0, 2.0, 1.5, 2.5])
        diagonal_energy = hamiltonian._calculate_diagonal_energy(0, 0)

        np.testing.assert_array_almost_equal(diagonal_energy, expected)

    def test_get_sho_potential(self) -> None:
        config: EigenstateConfig = {
            "mass": 1,
            "sho_omega": 1,
            "delta_x0": (2 * np.pi * hbar, 0),
            "delta_x1": (0, 2 * np.pi * hbar),
            "resolution": (2, 2, 2),
        }
        data: EnergyInterpolation = {
            "points": np.zeros(shape=(4, 4, 5)).tolist(),
            "dz": 1,
        }
        hamiltonian = SurfaceHamiltonianUtil(config, data, -2)
        expected = [2.0, 0.5, 0.0, 0.5, 2.0]
        np.testing.assert_equal(expected, hamiltonian.get_sho_potential())

    def test_get_sho_subtracted_points(self) -> None:
        nx = random.randrange(2, 20)
        ny = random.randrange(2, 20)
        nz = random.randrange(2, 100)

        z_offset = -20
        config: EigenstateConfig = {
            "mass": 1,
            "sho_omega": 1,
            "delta_x0": (2 * np.pi * hbar, 0),
            "delta_x1": (0, 2 * np.pi * hbar),
            "resolution": (nx, ny, 2),
        }
        data: EnergyInterpolation = {
            "points": np.zeros(shape=(2 * nx, 2 * ny, nz)).tolist(),
            "dz": 1,
        }
        hamiltonian = SurfaceHamiltonianUtil(config, data, z_offset)

        data2: EnergyInterpolation = {
            "points": np.tile(
                hamiltonian.get_sho_potential(), (2 * nx, 2 * ny, 1)
            ).tolist(),
            "dz": 1,
        }
        hamiltonian = SurfaceHamiltonianUtil(config, data2, z_offset)
        actual = hamiltonian.get_sho_subtracted_points()
        expected = np.zeros(shape=(2 * nx, 2 * ny, nz))

        np.testing.assert_allclose(expected, actual)

    def test_get_fft_is_real(self) -> None:
        width = random.randrange(1, 10) * 2
        nz = random.randrange(2, 100)

        points = generate_symmetrical_points(nz, width)
        config: EigenstateConfig = {
            "mass": 1,
            "sho_omega": 1,
            "delta_x0": (2 * np.pi * hbar, 0),
            "delta_x1": (0, 2 * np.pi * hbar),
            "resolution": (points.shape[0] // 2, points.shape[1] // 2, 2),
        }
        data: EnergyInterpolation = {
            "points": points.tolist(),
            "dz": 1,
        }
        hamiltonian = SurfaceHamiltonianUtil(config, data, -2)

        self.assertTrue(np.all(np.isreal(hamiltonian.get_ft_potential())))

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
            self.assertTrue(np.all(np.isclose(ft_potential[:, :, iz], ft_value)))

    def test_get_off_diagonal_energies_zero(self) -> None:
        hamiltonian = generate_random_diagonal_hamiltonian()

        actual = hamiltonian._calculate_off_diagonal_energies()
        n_points = hamiltonian.Nkx0 * hamiltonian.Nkx1 * hamiltonian.Nkz
        expected_shape = (n_points, n_points)
        np.testing.assert_equal(actual, np.zeros(shape=expected_shape))

    def test_is_almost_hermitian(self) -> None:
        width = random.randrange(1, 10) * 2
        nz = random.randrange(2, 100)

        points = generate_symmetrical_points(nz, width)
        np.testing.assert_allclose(points[1:, 1:], points[1:, 1:][::-1, ::-1])
        config: EigenstateConfig = {
            "mass": 1,
            "sho_omega": 1,
            "delta_x0": (2 * np.pi * hbar, 0),
            "delta_x1": (0, 2 * np.pi * hbar),
            "resolution": (width // 2, width // 2, 10),
        }
        data: EnergyInterpolation = {
            "points": points.tolist(),
            "dz": 1,
        }
        hamiltonian = SurfaceHamiltonianUtil(config, data, -2)

        np.testing.assert_allclose(
            hamiltonian.hamiltonian(0, 0), hamiltonian.hamiltonian(0, 0).conjugate().T
        )

    def test_sho_normalization(self) -> None:
        nx = random.randrange(2, 10)
        ny = random.randrange(2, 10)
        nz = 1001
        z_width = 20

        z_offset = -z_width / 2
        config: EigenstateConfig = {
            "mass": hbar**2,
            "sho_omega": 1 / hbar,
            "delta_x0": (2 * np.pi * hbar, 0),
            "delta_x1": (0, 2 * np.pi * hbar),
            "resolution": (nx, ny, 10),
        }
        data: EnergyInterpolation = {
            "points": np.zeros(shape=(2 * nx, 2 * ny, nz)).tolist(),
            "dz": z_width / (nz - 1),
        }
        hamiltonian = SurfaceHamiltonianUtil(config, data, z_offset)

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
        config: EigenstateConfig = {
            "mass": hbar**2,
            "sho_omega": 1 / hbar,
            "delta_x0": (2 * np.pi * hbar, 0),
            "delta_x1": (0, 2 * np.pi * hbar),
            "resolution": (nx // 2, ny // 2, 14),
        }
        data: EnergyInterpolation = {
            "points": np.zeros(shape=(nx, ny, nz)).tolist(),
            "dz": 1,
        }

        hamiltonian = SurfaceHamiltonianUtil(config, data, 0)

        np.testing.assert_allclose(
            hamiltonian._calculate_off_diagonal_energies_fast(),
            hamiltonian._calculate_off_diagonal_energies(),
        )

    def test_eigenstate_normalization(self) -> None:
        width = random.randrange(4, 20)
        nz = 100
        config: EigenstateConfig = {
            "mass": hbar**2,
            "sho_omega": 1 / hbar,
            "delta_x0": (2 * np.pi * hbar, 0),
            "delta_x1": (0, 2 * np.pi * hbar),
            "resolution": (width // 2, width // 2, 14),
        }

        points = generate_symmetrical_points(nz, width)
        data: EnergyInterpolation = {
            "points": points.tolist(),
            "dz": 1,
        }

        hamiltonian = SurfaceHamiltonianUtil(config, data, 0)

        kx = 0
        ky = 0
        eig_val, eig_states = hamiltonian.calculate_eigenvalues(kx, ky)

        np.testing.assert_allclose(
            np.array([np.linalg.norm(x["eigenvector"]) for x in eig_states]),
            np.ones_like(eig_val),
        )


if __name__ == "__main__":
    unittest.main()
