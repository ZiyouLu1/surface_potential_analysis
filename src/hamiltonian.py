import datetime
import math
from functools import cache, cached_property
from typing import Any, Tuple

import numpy as np
import scipy.special
from numpy.typing import NDArray
from scipy.constants import hbar

import hamiltonian_diag
from energy_data import EnergyInterpolation
from sho_config import SHOConfig


def calculate_sho_wavefunction(z_points, sho_omega, mass, n) -> NDArray:
    norm = (sho_omega * mass / hbar) ** 0.5
    normalized_z = z_points * norm

    prefactor = np.sqrt(norm / (2**n * math.factorial(n) * np.sqrt(np.pi)))
    hermite = scipy.special.eval_hermite(n, normalized_z)
    exponential = np.exp(-np.square(normalized_z) / 2)
    return prefactor * hermite * exponential


class SurfaceHamiltonian:

    _potential: EnergyInterpolation

    _config: SHOConfig

    _resolution: Tuple[int, int, int]

    def __init__(
        self,
        resolution: Tuple[int, int, int],
        potential: EnergyInterpolation,
        config: SHOConfig,
    ) -> None:

        self._potential = potential
        self._config = config
        self._resolution = resolution

        if (2 * self._resolution[0]) > self.Nx:
            print("Warning: max(ndkx) > Nx, some over sampling will occur")
        if (2 * self._resolution[1]) > self.Ny:
            print("Warning: max(ndky) > Ny, some over sampling will occur")

    @property
    def points(self):
        return np.array(self._potential["points"])

    @property
    def x_points(self):
        """
        Calculate the lattice coordinates in the x direction

        Note: We don't store the 'nth' pixel
        """
        return np.linspace(0, self.delta_x, self.Nx, endpoint=False)

    @property
    def y_points(self):
        """
        Calculate the lattice coordinates in the y direction

        Note: We don't store the 'nth' pixel
        """
        return np.linspace(0, self.delta_y, self.Ny, endpoint=False)

    @property
    def z_points(self):
        z_start = self.z_offset
        z_end = self.z_offset + (self.Nz - 1) * self.dz
        return np.linspace(z_start, z_end, self.Nz)

    @property
    def z_offset(self):
        return self._config["z_offset"]

    @property
    def mass(self):
        return self._config["mass"]

    @property
    def sho_omega(self):
        return self._config["sho_omega"]

    @property
    def delta_x(self) -> float:
        return self._potential["delta_x"]

    @cached_property
    def dkx(self) -> float:
        return 2 * np.pi / self.delta_x

    @property
    def Nx(self) -> int:
        return self.points.shape[0]

    @property
    def delta_y(self) -> float:
        return self._potential["delta_y"]

    @cached_property
    def dky(self) -> float:
        return 2 * np.pi / (self.delta_y)

    @property
    def Ny(self) -> int:
        return self.points.shape[1]

    @cached_property
    def dz(self) -> float:
        return self._potential["dz"]

    @property
    def Nz(self) -> int:
        return self.points.shape[2]

    @cached_property
    def coordinates(self) -> NDArray:
        return np.array([x for x in np.ndindex(*self._resolution)])

    @cached_property
    def hamiltonian(self) -> NDArray:
        t1 = datetime.datetime.now()
        diagonal_energies = np.diag(self._calculate_diagonal_energy())
        t2 = datetime.datetime.now()
        print((t2 - t1).total_seconds())
        other_energies = self._calculate_off_diagonal_energies_fast()
        print((datetime.datetime.now() - t2).total_seconds())

        energies = diagonal_energies + other_energies
        if not np.allclose(energies, energies.conjugate().T):
            raise AssertionError("hamiltonian is not hermitian")
        return energies

    _eigenvalues: None | NDArray = None
    _eigenvectors: None | NDArray = None

    @property
    def eigenvalues(self) -> NDArray:
        if self._eigenvalues is None:
            e, _ = self._calculate_eigenvalues()
            return e
        return self._eigenvalues

    @property
    def eigenvectors(self) -> NDArray:
        if self._eigenvectors is None:
            _, e = self._calculate_eigenvalues()
            return e
        return self._eigenvectors

    def _calculate_eigenvalues(self) -> Tuple[NDArray, NDArray]:
        w, v = np.linalg.eigh(self.hamiltonian)
        self._eigenvalues = w
        self._eigenvectors = v
        return (w, v)

    def _calculate_diagonal_energy(self) -> NDArray[Any]:

        x_coords, y_coords, z_coords = self.coordinates.T

        x_energy = (hbar * self.dkx * x_coords) ** 2 / (2 * self.mass)
        y_energy = (hbar * self.dky * y_coords) ** 2 / (2 * self.mass)
        z_energy = (hbar * self.sho_omega) * (z_coords + 0.5)
        return x_energy + y_energy + z_energy

    def get_index(self, nkx: int, nky: int, nz: int) -> int:
        ikx = nkx * self._resolution[1] * self._resolution[2]
        iky = nky * self._resolution[2]
        return ikx + iky + nz

    @cache
    def get_sho_potential(self) -> NDArray:
        return 0.5 * self.mass * self.sho_omega**2 * np.square(self.z_points)

    @cache
    def get_sho_subtracted_points(self) -> NDArray:
        return np.subtract(self.points, self.get_sho_potential())

    @cache
    def get_ft_potential(self) -> NDArray:
        subtracted_potential = self.get_sho_subtracted_points()
        fft_potential = np.fft.ifft2(subtracted_potential, axes=(0, 1))

        if not np.all(np.isreal(np.real_if_close(fft_potential))):
            raise AssertionError("FFT was not real!")
        return np.real_if_close(fft_potential)

    @cache
    def get_ft_potential_fast(self) -> NDArray:
        subtracted_potential = self.get_sho_subtracted_points()
        fft_potential = np.fft.ifft2(subtracted_potential, axes=(0, 1))
        return np.real_if_close(fft_potential)

    @cache
    def _calculate_sho_wavefunction_points(self, n: int) -> NDArray:
        """Generates the nth SHO wavefunction using the current config"""
        return calculate_sho_wavefunction(self.z_points, self.sho_omega, self.mass, n)

    @cache
    def _calculate_off_diagonal_entry(self, nz1, nz2, ndkx, ndky) -> float:
        """Calculates the off diagonal energy using the 'folded' points ndkx, ndky"""
        ft_pot_points = self.get_ft_potential()[ndkx, ndky]
        hermite1 = self._calculate_sho_wavefunction_points(nz1)
        hermite2 = self._calculate_sho_wavefunction_points(nz2)

        fourier_transform = np.sum(hermite1 * hermite2 * ft_pot_points)

        return self.dz * fourier_transform

    def _calculate_off_diagonal_energies_fast(self) -> NDArray:

        return np.array(
            hamiltonian_diag.get_hamiltonian(
                self.get_ft_potential_fast().tolist(),  # Takes abt 0.3s for a 10s run
                self._resolution,
                self.dz,
                self.mass,
                self.sho_omega,
                self.z_offset,
            )
        )

    def _calculate_off_diagonal_energies(self) -> NDArray:

        n_coordinates = len(self.coordinates)
        hamiltonian = np.zeros(shape=(n_coordinates, n_coordinates))

        for (index1, [nkx1, nky1, nz1]) in enumerate(self.coordinates):
            for (index2, [nkx2, nky2, nz2]) in enumerate(self.coordinates):
                # Number of jumps in units of dkx for this matrix element

                # As k-> k+ Nx * dkx the ft potential is left unchanged
                # Therefore we 'wrap round' ndkx into the region 0<ndkx<Nx
                # Where Nx is the number of x points

                # In reality we want to make sure ndkx < Nx (ie we should
                # make sure we generate enough points in the interpolation)
                ndkx = (nkx2 - nkx1) % self.Nx
                ndky = (nky2 - nky1) % self.Ny

                hamiltonian[index1, index2] = self._calculate_off_diagonal_entry(
                    nz1, nz2, ndkx, ndky
                )

        return hamiltonian

    def calculate_wavefunction(self, points: NDArray, eigenvector: NDArray) -> NDArray:
        out = np.zeros(shape=(points.shape[0]), dtype=complex)
        for (e, [nkx, nky, nz]) in zip(eigenvector, self.coordinates):
            out += (
                e
                * calculate_sho_wavefunction(
                    points[:, 2], self.sho_omega, self.mass, nz
                )
                * np.exp(1j * nkx * self.dkx * points[:, 0])
                * np.exp(1j * nky * self.dky * points[:, 1])
            )
        return out


if __name__ == "__main__":
    data: EnergyInterpolation = {
        "points": [[[0, 0], [0, 0]], [[0, 0], [0, 0]]],
        "delta_x": 2 * np.pi * hbar,
        "delta_y": 2 * np.pi * hbar,
        "dz": 1,
    }
    config: SHOConfig = {"mass": 1, "sho_omega": 1 / hbar, "z_offset": 0}
    hamiltonian = SurfaceHamiltonian((2, 2, 2), data, config)
