from functools import cached_property
from typing import TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.constants import hbar

import hamiltonian_generator
from surface_potential_analysis.sho_wavefunction import calculate_sho_wavefunction
from surface_potential_analysis.surface_config import (
    SurfaceConfig,
    SurfaceConfigUtil,
    get_surface_xy_points,
)


def make_vectors_orthogonal(vectors: list[NDArray]) -> list[NDArray]:
    out: list[NDArray] = []
    for v in vectors:
        v_out = v
        for v1 in out:
            v_out -= (np.dot(v_out, v1) * v1) / (np.linalg.norm(v1) ** 2)

        out.append(v_out)
    return out


class EigenstateConfig(SurfaceConfig):
    resolution: tuple[int, int, int]
    """Resolution in x,y,z to produce the eigenstates in"""
    sho_omega: float
    """Angular frequency (in rad s-1) of the sho we will fit using"""
    mass: float
    """Mass in Kg"""


class Eigenstate(TypedDict):
    kx: float
    ky: float
    eigenvector: list[complex]


def calculate_wavefunction_fast(
    config: EigenstateConfig, eigenstate: Eigenstate, points: ArrayLike
) -> NDArray:
    return np.array(
        hamiltonian_generator.get_eigenstate_wavefunction(
            config["resolution"],
            config["delta_x0"],
            config["delta_x1"],
            config["mass"],
            config["sho_omega"],
            eigenstate["kx"],
            eigenstate["ky"],
            eigenstate["eigenvector"],
            np.array(points).tolist(),
        ),
        dtype=complex,
    )


class EigenstateConfigUtil(SurfaceConfigUtil):
    _config: EigenstateConfig

    def __init__(self, config: EigenstateConfig) -> None:
        super().__init__(config)

    @property
    def resolution(self):
        return self._config["resolution"]

    @property
    def mass(self):
        return self._config["mass"]

    @property
    def sho_omega(self):
        return self._config["sho_omega"]

    @property
    def Nkx0(self) -> int:
        return self.resolution[0]

    @property
    def nkx0_points(self):
        return np.array(np.rint(np.fft.fftfreq(self.Nkx0, 1 / self.Nkx0)), dtype=int)

    @property
    def Nkx1(self) -> int:
        return self.resolution[1]

    @property
    def nkx1_points(self):
        return np.array(np.rint(np.fft.fftfreq(self.Nkx1, 1 / self.Nkx1)), dtype=int)

    @property
    def Nkz(self) -> int:
        return self.resolution[2]

    @property
    def nz_points(self):
        return np.arange(self.Nkz, dtype=int)

    @property
    def characteristic_z(self) -> float:
        """Get the characteristic Z length, given by sqrt(hbar / m * omega)"""
        return np.sqrt(hbar / (self.mass * self.sho_omega))

    @cached_property
    def eigenstate_indexes(self) -> NDArray:
        x0t, x1t, zt = np.meshgrid(
            self.nkx0_points, self.nkx1_points, self.nz_points, indexing="ij"
        )
        return np.array([x0t.ravel(), x1t.ravel(), zt.ravel()]).T

    def get_index(self, nkx0: int, nkx1: int, nz: int) -> int:
        ikx0 = (nkx0 % self.Nkx0) * self.Nkx1 * self.Nkz
        ikx1 = (nkx1 % self.Nkx1) * self.Nkz
        return ikx0 + ikx1 + nz

    def calculate_wavefunction_slow(
        self,
        eigenstate: Eigenstate,
        points: ArrayLike,
        cutoff: int | None = None,
    ) -> NDArray:
        points = np.array(points)
        out = np.zeros(shape=(points.shape[0]), dtype=complex)

        eigenvector_array = np.array(eigenstate["eigenvector"])
        coordinates = self.eigenstate_indexes
        args = (
            np.arange(self.eigenstate_indexes.shape[0])
            if cutoff is None
            else np.argsort(np.abs(eigenvector_array))[::-1][:cutoff]
        )
        kx = eigenstate["kx"]
        ky = eigenstate["ky"]
        for arg in args:
            (nkx0, nkx1, nz) = coordinates[arg]
            e = eigenvector_array[arg]
            x_phase = (nkx0 * self.dkx0[0] + nkx1 * self.dkx1[0] + kx) * points[:, 0]
            y_phase = (nkx0 * self.dkx0[1] + nkx1 * self.dkx1[1] + ky) * points[:, 1]
            out += (
                e
                * calculate_sho_wavefunction(
                    points[:, 2], self.sho_omega, self.mass, nz
                )
                * np.exp(1j * (x_phase + y_phase))
            )
        return out

    def calculate_bloch_wavefunction_fourier(
        self, eigenvector: list[complex], z_points: list[float]
    ) -> NDArray:
        """
        Calculates the bloch wavefunction at the fundamental points in the unit cell
        (ie the points we have information on based off of the information given in the eigenvector)

        Note we use the convention that (ignoring the ikx1 direction)

        :math:`\\psi(x) = sum_{ikx0} A_{ikx0} \\exp{2 \\pi (ikx0) * dkx0 * x}`

        Parameters
        ----------
        eigenvector : list[complex]
        z_points    : list[float]


        Returns
        -------
        NDArray
            An array containing the bloch wavefunction amplitudes as a list[list[list[complex]]]
            of amplitudes for each ix, iy, iz

        """
        sho_wavefunctions = np.array(
            make_vectors_orthogonal(
                [
                    calculate_sho_wavefunction(z_points, self.sho_omega, self.mass, nz)
                    for nz in range(self.Nkz)
                ]
            )
        )

        # List, list, list of amplitudes for each ikx, iky, ikz
        eigenvector_array = np.reshape(eigenvector, (self.Nkx0, self.Nkx1, self.Nkz))

        # List, list, list of amplitudes for each ix, iy, ikz
        # Use the "forward" norm to prevent the extra factor of 1/nx, 1/ny here
        ft_points = np.fft.ifft2(eigenvector_array, axes=(0, 1), norm="forward")

        # List, list, list, list of amplitudes for each ix, iy, ikz, iz
        points_each_iz = np.multiply(
            ft_points[:, :, :, np.newaxis],
            sho_wavefunctions[np.newaxis, np.newaxis, :, :],
        )
        # Sum over sho wavefunctions for each ikz
        bloch_points = np.sum(points_each_iz, axis=2)

        return bloch_points

    def _get_fourier_coordinates_in_grid(
        self, x0_lim: tuple[int, int] = (0, 1), x1_lim: tuple[int, int] = (0, 1)
    ) -> NDArray:
        nx0 = x0_lim[1] - x0_lim[0]
        nx1 = x1_lim[1] - x1_lim[0]
        # Create a fake surface config containing the whole region in a single unit cell
        xy_shape = (self.Nkx0 * nx0, self.Nkx1 * nx1)
        config: SurfaceConfig = {
            "delta_x0": (
                nx0 * self.delta_x0[0],
                nx0 * self.delta_x0[1],
            ),
            "delta_x1": (
                nx1 * self.delta_x1[0],
                nx1 * self.delta_x1[1],
            ),
        }
        return get_surface_xy_points(
            config,
            xy_shape,
            offset=(
                x0_lim[0] * self.delta_x0[0] + x1_lim[0] * self.delta_x1[0],
                x0_lim[0] * self.delta_x0[1] + x1_lim[0] * self.delta_x1[1],
            ),
        )

    def calculate_wavefunction_slow_grid_fourier_exact_phase(
        self,
        eigenvector: list[complex],
        ns: tuple[int, int],
        Ns: tuple[int, int],
        z_points: list[float],
        x0_lim: tuple[int, int] = (0, 1),
        x1_lim: tuple[int, int] = (0, 1),
    ) -> NDArray:
        """
        Parameters
        ----------
        eigenvector: list[float]
        ns         : tuple[int, int]
            Index of the current sample
        Ns         : tuple[int, int]
            Total resolution of sample
        z_points   : list[float]
        x0_lim     : tuple[int, int], optional
            Region to sample the wavefunction in the x0 direction, exclusive on the second argument.
        x0_lim     : tuple[int, int], optional
            Region to sample the wavefunction in the x1 direction, exclusive on the second argument.


        Returns
        -------
        NDArray
            An array containing the wavefunction amplitudes as a list[list[list[complex]]]
            of amplitudes for each ix, iy, iz

        """

        x0v, x1v = np.meshgrid(
            np.arange(self.Nkx0 * x0_lim[0], self.Nkx0 * x0_lim[1]),
            np.arange(self.Nkx1 * x1_lim[0], self.Nkx1 * x1_lim[1]),
            indexing="ij",
        )

        phases = (ns[0] * x0v / (self.Nkx0 * Ns[0])) + (
            ns[1] * x1v / (self.Nkx1 * Ns[1])
        )
        phase_points = np.exp(2j * np.pi * (phases))

        # Bloch wavevector ignoring overall phase
        bloch_points = self.calculate_bloch_wavefunction_fourier(eigenvector, z_points)
        # The bloch wavefunctions (by definition) repeat in each unit cell
        nx0 = x0_lim[1] - x0_lim[0]
        nx1 = x1_lim[1] - x1_lim[0]
        repeated_bloch_points = np.tile(bloch_points, (nx0, nx1, 1))

        # Multiply each x, y point by the corresponding phase
        return repeated_bloch_points * phase_points[:, :, np.newaxis]

    def calculate_wavefunction_slow_grid_fourier(
        self,
        eigenstate: Eigenstate,
        z_points: list[float],
        x0_lim: tuple[int, int] = (0, 1),
        x1_lim: tuple[int, int] = (0, 1),
    ) -> NDArray:
        """
        Parameters
        ----------
        eigenstate : Eigenstate
        z_points   : list[float]
        x0_lim     : tuple[int, int], optional
            Region to sample the wavefunction in the x0 direction, exclusive on the second argument.
        x0_lim     : tuple[int, int], optional
            Region to sample the wavefunction in the x1 direction, exclusive on the second argument.


        Returns
        -------
        NDArray
            An array containing the wavefunction amplitudes as a list[list[list[complex]]]
            of amplitudes for each ix, iy, iz

        """

        xy_points = self._get_fourier_coordinates_in_grid(x0_lim, x1_lim)

        kx = eigenstate["kx"]
        ky = eigenstate["ky"]
        phase_points = np.exp(1j * (xy_points[:, :, 0] * kx + xy_points[:, :, 1] * ky))

        # Bloch wavevector ignoring overall phase
        bloch_points = self.calculate_bloch_wavefunction_fourier(
            eigenstate["eigenvector"], z_points
        )
        # The bloch wavefunctions (by definition) repeat in each unit cell
        nx0 = x0_lim[1] - x0_lim[0]
        nx1 = x1_lim[1] - x1_lim[0]
        repeated_bloch_points = np.tile(bloch_points, (nx0, nx1, 1))

        # Multiply each x, y point by the corresponding phase
        return repeated_bloch_points * phase_points[:, :, np.newaxis]

    def calculate_wavefunction_fast(
        self,
        eigenstate: Eigenstate,
        points: ArrayLike,
    ) -> NDArray:
        return calculate_wavefunction_fast(self._config, eigenstate, points)
