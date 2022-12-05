import random
import unittest

import numpy as np
import scipy.linalg
import scipy.special
from scipy.constants import hbar

import hamiltonian_diag
from energy_data import EnergyInterpolation
from hamiltonian import SurfaceHamiltonian, calculate_sho_wavefunction
from sho_config import SHOConfig


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


def generate_random_diagonal_hamiltonian():
    nx = random.randrange(2, 10)
    ny = random.randrange(2, 10)
    nz = random.randrange(2, 100)

    nkx = random.randrange(2, 10)
    nky = random.randrange(2, 10)
    nkz = random.randrange(2, 10)
    resolution = (nkx, nky, nkz)
    config: SHOConfig = {
        "mass": 1,
        "sho_omega": 1,
        "z_offset": 20 * random.random(),
    }
    data: EnergyInterpolation = {
        "points": np.zeros(shape=(nx, ny, nz)).tolist(),
        "delta_x": 2 * np.pi * hbar,
        "delta_y": 2 * np.pi * hbar,
        "dz": 1,
    }
    hamiltonian = SurfaceHamiltonian(resolution, data, config)

    data2: EnergyInterpolation = {
        "points": np.tile(hamiltonian.get_sho_potential(), (nx, ny, 1)).tolist(),
        "delta_x": 2 * np.pi * hbar,
        "delta_y": 2 * np.pi * hbar,
        "dz": 1,
    }
    return SurfaceHamiltonian(resolution, data2, config)


class TestSurfaceHamiltonian(unittest.TestCase):
    def test_diagonal_energies(self) -> None:
        config: SHOConfig = {
            "mass": 1,
            "sho_omega": 1 / hbar,
            "z_offset": 0,
        }
        data: EnergyInterpolation = {
            "points": [[[0, 0], [0, 0]], [[0, 0], [0, 0]]],
            "delta_x": 2 * np.pi * hbar,
            "delta_y": 2 * np.pi * hbar,
            "dz": 1,
        }
        hamiltonian = SurfaceHamiltonian((2, 2, 2), data, config)

        expected = np.array([0.5, 1.5, 1.0, 2.0, 1.0, 2.0, 1.5, 2.5])
        diagonal_energy = hamiltonian._calculate_diagonal_energy()

        np.testing.assert_equal(diagonal_energy, expected)

    def test_get_all_coordinates(self) -> None:
        nkx = random.randrange(1, 20)
        nky = random.randrange(1, 20)
        nz = random.randrange(1, 100)

        xt, yt, zt = np.meshgrid(range(nkx), range(nky), range(nz), indexing="ij")
        expected = np.array([xt.ravel(), yt.ravel(), zt.ravel()]).T
        config: SHOConfig = {
            "mass": 1,
            "sho_omega": 1 / hbar,
            "z_offset": 0,
        }
        data: EnergyInterpolation = {
            "points": [[[0, 0], [0, 0]], [[0, 0], [0, 0]]],
            "delta_x": 2 * np.pi * hbar,
            "delta_y": 2 * np.pi * hbar,
            "dz": 1,
        }
        hamiltonian = SurfaceHamiltonian((nkx, nky, nz), data, config)
        coords = hamiltonian.coordinates
        for (e, a) in zip(expected, coords):
            self.assertEqual(e[0], a[0])
            self.assertEqual(e[1], a[1])
            self.assertEqual(e[2], a[2])
            self.assertEqual(3, len(a))

    def test_get_index(self) -> None:
        nkx = random.randrange(1, 20)
        nky = random.randrange(1, 20)
        nz = random.randrange(1, 100)

        config: SHOConfig = {
            "mass": 1,
            "sho_omega": 1 / hbar,
            "z_offset": 0,
        }
        data: EnergyInterpolation = {
            "points": [[[0, 0], [0, 0]], [[0, 0], [0, 0]]],
            "delta_x": 2 * np.pi * hbar,
            "delta_y": 2 * np.pi * hbar,
            "dz": 1,
        }
        hamiltonian = SurfaceHamiltonian((nkx, nky, nz), data, config)

        coords = hamiltonian.coordinates
        for (i, c) in enumerate(coords):
            self.assertEqual(i, hamiltonian.get_index(*c))

    def test_get_sho_potential(self) -> None:
        config: SHOConfig = {
            "mass": 1,
            "sho_omega": 1,
            "z_offset": -2,
        }
        data: EnergyInterpolation = {
            "points": np.zeros(shape=(2, 2, 5)).tolist(),
            "delta_x": 2 * np.pi * hbar,
            "delta_y": 2 * np.pi * hbar,
            "dz": 1,
        }
        hamiltonian = SurfaceHamiltonian((2, 2, 2), data, config)
        expected = [2.0, 0.5, 0.0, 0.5, 2.0]
        np.testing.assert_equal(expected, hamiltonian.get_sho_potential())

    def test_get_sho_subtracted_points(self) -> None:
        nx = random.randrange(2, 20)
        ny = random.randrange(2, 20)
        nz = random.randrange(2, 100)

        config: SHOConfig = {
            "mass": 1,
            "sho_omega": 1,
            "z_offset": -20,
        }
        data: EnergyInterpolation = {
            "points": np.zeros(shape=(nx, ny, nz)).tolist(),
            "delta_x": 2 * np.pi * hbar,
            "delta_y": 2 * np.pi * hbar,
            "dz": 1,
        }
        hamiltonian = SurfaceHamiltonian((2, 2, 2), data, config)

        data2: EnergyInterpolation = {
            "points": np.tile(hamiltonian.get_sho_potential(), (nx, ny, 1)).tolist(),
            "delta_x": 2 * np.pi * hbar,
            "delta_y": 2 * np.pi * hbar,
            "dz": 1,
        }
        hamiltonian = SurfaceHamiltonian((2, 2, 2), data2, config)
        actual = hamiltonian.get_sho_subtracted_points()
        expected = np.zeros(shape=(nx, ny, nz))

        np.testing.assert_allclose(expected, actual)

    def test_delta_x(self):
        nx = random.randrange(2, 10) * 2
        ny = random.randrange(2, 10) * 2
        x_points = np.linspace(0, 2 * np.pi * hbar, num=nx)
        y_points = np.linspace(0, 2 * np.pi * hbar, num=ny)
        config: SHOConfig = {
            "mass": 1,
            "sho_omega": 1,
            "z_offset": -2,
        }
        data: EnergyInterpolation = {
            "points": np.zeros(shape=(nx - 1, ny - 1, 5)).tolist(),
            "delta_x": x_points[-1],
            "delta_y": y_points[-1],
            "dz": 1,
        }
        hamiltonian = SurfaceHamiltonian((2, 2, 2), data, config)

        self.assertAlmostEqual(x_points[-1], hamiltonian.delta_x)
        self.assertAlmostEqual(y_points[-1], hamiltonian.delta_y)

    def test_get_fft_is_real(self) -> None:
        width = random.randrange(1, 10) * 2
        nz = random.randrange(2, 100)

        points = generate_symmetrical_points(nz, width)
        config: SHOConfig = {
            "mass": 1,
            "sho_omega": 1,
            "z_offset": -2,
        }
        data: EnergyInterpolation = {
            "points": points.tolist(),
            "delta_x": 2 * np.pi * hbar,
            "delta_y": 2 * np.pi * hbar,
            "dz": 1,
        }
        hamiltonian = SurfaceHamiltonian((2, 2, 2), data, config)

        self.assertTrue(np.all(np.isreal(hamiltonian.get_ft_potential())))

    def test_get_fft_fast(self) -> None:
        width = random.randrange(1, 10) * 2
        nz = random.randrange(2, 100)

        points = generate_symmetrical_points(nz, width)
        config: SHOConfig = {
            "mass": 1,
            "sho_omega": 1,
            "z_offset": -2,
        }
        data: EnergyInterpolation = {
            "points": points.tolist(),
            "delta_x": 2 * np.pi * hbar,
            "delta_y": 2 * np.pi * hbar,
            "dz": 1,
        }
        hamiltonian = SurfaceHamiltonian((2, 2, 2), data, config)

        np.testing.assert_allclose(
            hamiltonian.get_ft_potential(), hamiltonian.get_ft_potential_fast()
        )

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
        resolution = hamiltonian._resolution

        actual = hamiltonian._calculate_off_diagonal_energies()
        expected_shape = (np.prod(resolution), np.prod(resolution))
        np.testing.assert_equal(actual, np.zeros(shape=expected_shape))

    def test_is_almost_hermitian(self) -> None:
        width = random.randrange(1, 10) * 2
        nz = random.randrange(2, 100)

        points = generate_symmetrical_points(nz, width)
        np.testing.assert_allclose(points[1:, 1:], points[1:, 1:][::-1, ::-1])
        config: SHOConfig = {
            "mass": 1,
            "sho_omega": 1,
            "z_offset": -2,
        }
        data: EnergyInterpolation = {
            "points": points.tolist(),
            "delta_x": 2 * np.pi * hbar,
            "delta_y": 2 * np.pi * hbar,
            "dz": 1,
        }
        hamiltonian = SurfaceHamiltonian((10, 10, 10), data, config)

        np.testing.assert_allclose(
            hamiltonian.hamiltonian, hamiltonian.hamiltonian.conjugate().T
        )

    def test_get_eigenvalues_diag(self) -> None:
        hamiltonian = generate_random_diagonal_hamiltonian()
        print(hamiltonian.hamiltonian)
        print(hamiltonian._eigenvalues)

    def test_calculate_sho_wavefunction(self) -> None:
        mass = hbar**2
        sho_omega = 1 / hbar
        z_points = np.linspace(-10, 10, 11)

        norm = np.sqrt(mass * sho_omega / hbar)

        phi_0_norm = np.sqrt(norm / np.sqrt(np.pi))
        phi_0_expected = phi_0_norm * np.exp(-((z_points * norm) ** 2) / 2)
        phi_0_actual = calculate_sho_wavefunction(z_points, sho_omega, mass, 0)

        np.testing.assert_allclose(phi_0_expected, phi_0_actual)

        phi_1_norm = np.sqrt(2 * norm / np.sqrt(np.pi))
        phi_1_expected = phi_1_norm * z_points * np.exp(-((z_points * norm) ** 2) / 2)
        phi_1_actual = calculate_sho_wavefunction(z_points, sho_omega, mass, 1)

        np.testing.assert_allclose(phi_1_expected, phi_1_actual)

        phi_2_norm = np.sqrt(norm / (2 * np.sqrt(np.pi)))
        phi_2_poly = (2 * z_points**2 - 1) * np.exp(-((z_points * norm) ** 2) / 2)
        phi_2_expected = phi_2_norm * phi_2_poly
        phi_2_actual = calculate_sho_wavefunction(z_points, sho_omega, mass, 2)

        np.testing.assert_allclose(phi_2_expected, phi_2_actual)

        phi_3_norm = np.sqrt(norm / (3 * np.sqrt(np.pi)))
        phi_3_poly = (2 * z_points**3 - 3 * z_points) * np.exp(
            -((z_points * norm) ** 2) / 2
        )
        phi_3_expected = phi_3_norm * phi_3_poly
        phi_3_actual = calculate_sho_wavefunction(z_points, sho_omega, mass, 3)

        np.testing.assert_allclose(phi_3_expected, phi_3_actual)

    def test_sho_normalization(self) -> None:
        nx = random.randrange(2, 10)
        ny = random.randrange(2, 10)
        nz = 1001
        z_width = 20

        config: SHOConfig = {
            "mass": hbar**2,
            "sho_omega": 1 / hbar,
            "z_offset": -z_width / 2,
        }
        data: EnergyInterpolation = {
            "points": np.zeros(shape=(nx, ny, nz)).tolist(),
            "delta_x": 2 * np.pi * hbar,
            "delta_y": 2 * np.pi * hbar,
            "dz": z_width / (nz - 1),
        }
        hamiltonian = SurfaceHamiltonian((1, 1, 10), data, config)

        for iz1 in range(10):
            for iz2 in range(10):
                sho_1 = hamiltonian._calculate_sho_wavefunction_points(iz1)
                sho_2 = hamiltonian._calculate_sho_wavefunction_points(iz2)
                sho_norm = hamiltonian.dz * np.sum(sho_1 * sho_2)

                if iz1 == iz2:
                    self.assertAlmostEqual(sho_norm, 1.0)
                else:
                    self.assertAlmostEqual(sho_norm, 0.0)

    def test_get_sho_rust(self) -> None:

        mass = hbar**2 * random.random()
        sho_omega = random.random() / hbar
        z_points = np.linspace(-20 * random.random(), 20 * random.random(), 1000)

        for n in range(10):
            actual = hamiltonian_diag.get_sho_wavefunction(
                z_points.tolist(), sho_omega, mass, n
            )
            expected = calculate_sho_wavefunction(z_points, sho_omega, mass, n)

            np.testing.assert_allclose(actual, expected)

    def test_get_hermite_val_rust(self) -> None:
        n = random.randrange(1, 10)
        x = random.random() * 10 - 5
        self.assertAlmostEqual(
            hamiltonian_diag.get_hermite_val(x, n),
            scipy.special.eval_hermite(n, x),
            places=6,
        )

    def test_calculate_off_diagonal_energies_rust(self) -> None:
        nx = random.randrange(2, 10)
        ny = random.randrange(2, 10)
        nz = 100
        config: SHOConfig = {
            "mass": hbar**2,
            "sho_omega": 1 / hbar,
            "z_offset": 0,
        }
        data: EnergyInterpolation = {
            "points": np.zeros(shape=(nx, ny, nz)).tolist(),
            "delta_x": 2 * np.pi * hbar,
            "delta_y": 2 * np.pi * hbar,
            "dz": 1,
        }

        hamiltonian = SurfaceHamiltonian((2, 2, 3), data, config)

        np.testing.assert_allclose(
            hamiltonian._calculate_off_diagonal_energies_fast(),
            hamiltonian._calculate_off_diagonal_energies(),
        )
