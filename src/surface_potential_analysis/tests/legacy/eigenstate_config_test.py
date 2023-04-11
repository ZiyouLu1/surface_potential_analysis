import math
import random
import unittest

import hamiltonian_generator
import numpy as np
from scipy.constants import hbar

from surface_potential_analysis._legacy.eigenstate import Eigenstate
from surface_potential_analysis._legacy.energy_eigenstate import (
    EigenstateConfig,
    EigenstateConfigUtil,
    get_brillouin_points_irreducible_config,
)
from surface_potential_analysis._legacy.sho_wavefunction import (
    calculate_sho_wavefunction,
)
from surface_potential_analysis._legacy.surface_config import get_surface_coordinates


def generate_eigenstates_grid_points_100(
    config: EigenstateConfig, *, grid_size=8, include_zero=True
):
    util = EigenstateConfigUtil(config)
    dkx = util.dkx0[0]
    (kx_points, kx_step) = np.linspace(
        -dkx / 2, dkx / 2, 2 * grid_size, endpoint=False, retstep=True
    )
    dky = util.dkx1[1]
    (ky_points, ky_step) = np.linspace(
        -dky / 2, dky / 2, 2 * grid_size, endpoint=False, retstep=True
    )
    if not include_zero:
        kx_points += kx_step / 2
        ky_points += ky_step / 2

    xv, yv = np.meshgrid(kx_points, ky_points, indexing="ij")
    k_points = np.array([xv.ravel(), yv.ravel()]).T
    return k_points


class TestEigenstateConfig(unittest.TestCase):
    def test_inverse_lattuice_points_100(self) -> None:
        config: EigenstateConfig = {
            "mass": 1,
            "resolution": (1, 1, 1),
            "sho_omega": 1,
            "delta_x0": (1, 0),
            "delta_x1": (0, 2),
        }
        util = EigenstateConfigUtil(config)

        self.assertEqual(util.dkx0[0], 2 * np.pi)
        self.assertEqual(util.dkx0[1], 0)
        self.assertEqual(util.dkx1[0], 0)
        self.assertEqual(util.dkx1[1], np.pi)

    def test_inverse_lattuice_points_111(self) -> None:
        config: EigenstateConfig = {
            "mass": 1,
            "resolution": (1, 1, 1),
            "sho_omega": 1,
            "delta_x0": (1, 0),
            "delta_x1": (0.5, np.sqrt(3) / 2),
        }
        util = EigenstateConfigUtil(config)

        self.assertEqual(util.dkx0[0], 2 * np.pi)
        self.assertEqual(util.dkx0[1], -2 * np.pi / np.sqrt(3))
        self.assertEqual(util.dkx1[0], 0)
        self.assertEqual(util.dkx1[1], 4 * np.pi / np.sqrt(3))

    def test_get_kx_points(self) -> None:
        Nkx = random.randrange(1, 20)
        Nky = random.randrange(1, 20)
        Nz = random.randrange(1, 100)

        config: EigenstateConfig = {
            "mass": 1,
            "sho_omega": 1 / hbar,
            "delta_x0": (2 * np.pi * hbar, 0),
            "delta_x1": (0, 2 * np.pi * hbar),
            "resolution": (Nkx, Nky, Nz),
        }
        util = EigenstateConfigUtil(config)

        nkx_points = util.nkx0_points
        expected = np.concatenate(
            (np.arange(0, math.ceil(Nkx / 2)), np.arange(-math.floor(Nkx / 2), 0))
        )
        np.testing.assert_array_equal(nkx_points, expected)

        nky_points = util.nkx1_points
        expected = np.concatenate(
            (np.arange(0, math.ceil(Nky / 2)), np.arange(-math.floor(Nky / 2), 0))
        )
        np.testing.assert_array_equal(nky_points, expected)

    def test_get_all_coordinates(self) -> None:
        Nkx = random.randrange(1, 40)
        Nky = random.randrange(1, 40)
        Nkz = random.randrange(1, 25)

        config: EigenstateConfig = {
            "mass": 1,
            "sho_omega": 1 / hbar,
            "delta_x0": (2 * np.pi * hbar, 0),
            "delta_x1": (0, 2 * np.pi * hbar),
            "resolution": (Nkx, Nky, Nkz),
        }
        util = EigenstateConfigUtil(config)
        coords = util.eigenstate_indexes
        for i, (nkx, nky, nz) in enumerate(coords):
            self.assertEqual(util.get_index(nkx, nky, nz), i)

    def test_get_index(self) -> None:
        Nkx = random.randrange(1, 40)
        Nky = random.randrange(1, 40)
        Nkz = random.randrange(1, 25)

        config: EigenstateConfig = {
            "mass": 1,
            "sho_omega": 1 / hbar,
            "delta_x0": (2 * np.pi * hbar, 0),
            "delta_x1": (0, 2 * np.pi * hbar),
            "resolution": (Nkx, Nky, Nkz),
        }
        util = EigenstateConfigUtil(config)

        coords = util.eigenstate_indexes
        for i, c in enumerate(coords):
            self.assertEqual(i, util.get_index(*c))

    def test_delta_x(self) -> None:
        nx = random.randrange(2, 10) * 2
        ny = random.randrange(2, 10) * 2
        x_points = np.linspace(0, 2 * np.pi * hbar, num=nx)
        y_points = np.linspace(0, 2 * np.pi * hbar, num=ny)
        config: EigenstateConfig = {
            "mass": 1,
            "sho_omega": 1,
            "delta_x0": (x_points[-1], 0),
            "delta_x1": (0, y_points[-1]),
            "resolution": (2, 2, 2),
        }
        hamiltonian = EigenstateConfigUtil(config)

        self.assertAlmostEqual(x_points[-1], hamiltonian.delta_x0[0])
        self.assertAlmostEqual(y_points[-1], hamiltonian.delta_x1[0])

    def test_calculate_wavefunction_fast(self) -> None:
        config: EigenstateConfig = {
            "mass": hbar**2,
            "sho_omega": 1 / hbar,
            "delta_x0": (2 * np.pi, 0),
            "delta_x1": (0, 2 * np.pi),
            "resolution": (10, 10, 14),
        }

        util = EigenstateConfigUtil(config)
        kx = np.random.rand() * (np.abs(util.dkx0[0]) + np.abs(util.dkx1[0])) / 2
        ky = np.random.rand() * (np.abs(util.dkx0[1]) + np.abs(util.dkx1[1])) / 2

        sample = np.random.normal(size=util.eigenstate_indexes.shape[0])
        eigenvector = (sample / np.linalg.norm(sample)).tolist()
        points = [[1, 1, 1]]

        expected = util.calculate_wavefunction_slow(
            {"eigenvector": eigenvector, "kx": kx, "ky": ky}, points
        )
        actual = util.calculate_wavefunction_fast(
            {"eigenvector": eigenvector, "kx": kx, "ky": ky}, points
        )

        np.testing.assert_allclose(expected, actual)

    def test_calculate_wavefunction_fourier(self) -> None:
        config: EigenstateConfig = {
            "mass": hbar**2,
            "sho_omega": 1 / hbar,
            "delta_x0": (2 * np.pi, 0),
            "delta_x1": (0, 2 * np.pi),
            "resolution": (10, 10, 14),
        }
        util = EigenstateConfigUtil(config)
        kx = np.random.rand() * (np.abs(util.dkx0[0]) + np.abs(util.dkx1[0])) / 2
        ky = np.random.rand() * (np.abs(util.dkx0[1]) + np.abs(util.dkx1[1])) / 2

        sample = np.random.normal(size=util.eigenstate_indexes.shape[0])
        eigenvector = (sample / np.linalg.norm(sample)).tolist()

        z_points = [0.0]
        points = get_surface_coordinates(
            config, (config["resolution"][0], config["resolution"][1]), z_points
        ).reshape(-1, 3)

        actual = util.calculate_wavefunction_slow_grid_fourier(
            {"eigenvector": eigenvector, "kx": kx, "ky": ky}, z_points
        )
        expected = util.calculate_wavefunction_slow(
            {"eigenvector": eigenvector, "kx": kx, "ky": ky}, points
        ).reshape(actual.shape)

        np.testing.assert_allclose(expected, actual)

    def test_generate_brillouin_zone_points_copper(self) -> None:
        config: EigenstateConfig = {
            "mass": hbar**2,
            "sho_omega": 1 / hbar,
            "delta_x0": (2 * np.pi * hbar, 0),
            "delta_x1": (0, 2 * np.pi * hbar),
            "resolution": (10, 10, 14),
        }
        expected = generate_eigenstates_grid_points_100(config, grid_size=4)
        actual = get_brillouin_points_irreducible_config(config, size=(4, 4))
        np.testing.assert_allclose(expected, actual)

    def test_eigenstate_periodicity(self) -> None:
        config: EigenstateConfig = {
            "mass": hbar**2,
            "sho_omega": 1 / hbar,
            "delta_x0": (2 * np.pi * hbar, 0),
            "delta_x1": (0, 2 * np.pi * hbar),
            "resolution": (1, 1, 14),
        }

        util = EigenstateConfigUtil(config)

        kx = np.random.uniform(low=-util.dkx0[0] / 2, high=util.dkx0[0] / 2)
        ky = np.random.uniform(low=-util.dkx1[1] / 2, high=util.dkx1[1] / 2)
        eigenvector = np.random.rand(util.eigenstate_indexes.shape[0]).tolist()

        x = np.random.uniform(low=0.0, high=util.delta_x0[0], size=100)
        y = np.random.uniform(low=0.0, high=util.delta_x1[1], size=100)

        center = util.calculate_wavefunction_fast(
            {"kx": kx, "ky": ky, "eigenvector": eigenvector},
            np.array([x, y, np.zeros_like(x)]).T.tolist(),
        )
        x_offset = util.calculate_wavefunction_fast(
            {"kx": kx, "ky": ky, "eigenvector": eigenvector},
            np.array([x + util.delta_x0[0], y, np.zeros_like(x)]).T.tolist(),
        )
        y_offset = util.calculate_wavefunction_fast(
            {"kx": kx, "ky": ky, "eigenvector": eigenvector},
            np.array([x, y + util.delta_x1[1], np.zeros_like(x)]).T.tolist(),
        )
        np.testing.assert_allclose(
            center, x_offset * np.exp(-1j * kx * util.delta_x0[0])
        )
        np.testing.assert_allclose(
            center, y_offset * np.exp(-1j * ky * util.delta_x1[1])
        )

    def test_wavefunction_normalization_z(self) -> None:
        Nkx = np.random.randint(1, 10)
        Nky = np.random.randint(1, 10)
        config: EigenstateConfig = {
            "mass": hbar**2,
            "sho_omega": 1 / hbar,
            "delta_x0": (2 * np.pi * hbar, 0),
            "delta_x1": (0, 2 * np.pi * hbar),
            "resolution": (Nkx, Nky, 14),
        }

        util = EigenstateConfigUtil(config)
        for iz in range(util.Nkz):
            eigenvector = np.zeros(shape=util.eigenstate_indexes.shape[0])
            eigenvector[util.get_index(0, 0, iz)] = 1
            eigenstate: Eigenstate = {
                "eigenvector": eigenvector.tolist(),
                "kx": 0,
                "ky": 0,
            }
            delta_z = 16 * util.characteristic_z
            n = 2000
            z_points = np.linspace(-delta_z / 2, delta_z / 2, n)
            points = [[0, 0, z] for z in z_points]
            wavefunction = util.calculate_wavefunction_fast(eigenstate, points)
            # note dz = delta_z / n
            norm = np.sum(np.square(np.abs(wavefunction))) * delta_z / n

            self.assertAlmostEqual(1, norm, places=2, msg=f"failed for iz={iz}")

    def test_wavefunction_normalization_xyz(self) -> None:
        Nkx = np.random.randint(1, 10)
        Nky = np.random.randint(1, 10)
        config: EigenstateConfig = {
            "mass": hbar**2,
            "sho_omega": 1 / hbar,
            "delta_x0": (2 * np.pi * hbar, 0),
            "delta_x1": (0, 2 * np.pi * hbar),
            "resolution": (Nkx, Nky, 14),
        }

        util = EigenstateConfigUtil(config)
        for iz in range(util.Nkz):
            eigenvector = np.zeros(shape=util.eigenstate_indexes.shape[0])
            eigenvector[util.get_index(0, 0, iz)] = 1
            eigenstate: Eigenstate = {
                "eigenvector": eigenvector.tolist(),
                "kx": 0,
                "ky": 0,
            }
            delta_z = 16 * util.characteristic_z
            nz = 10000
            z_points = np.linspace(-delta_z / 2, delta_z / 2, nz)
            wavefunction = util.calculate_wavefunction_slow_grid_fourier(
                eigenstate, z_points
            )
            nx = wavefunction.shape[0]
            ny = wavefunction.shape[1]
            n = nx * ny * nz
            # note dz = delta_z / n
            norm = np.sum(np.square(np.abs(wavefunction))) * delta_z / n

            self.assertAlmostEqual(1, norm, places=3, msg=f"failed for iz={iz}")
