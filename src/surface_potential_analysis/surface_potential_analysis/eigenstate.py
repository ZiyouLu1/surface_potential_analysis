from functools import cached_property
from typing import List, Tuple, TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.constants import hbar

import hamiltonian_generator
from surface_potential_analysis.sho_wavefunction import calculate_sho_wavefunction
from surface_potential_analysis.surface_config import SurfaceConfig, SurfaceConfigUtil


class EigenstateConfig(SurfaceConfig):
    resolution: Tuple[int, int, int]
    """Resolution in x,y,z to produce the eigenstates in"""
    sho_omega: float
    """Angular frequency (in rad s-1) of the sho we will fit using"""
    mass: float
    """Mass in Kg"""


class Eigenstate(TypedDict):
    kx: float
    ky: float
    eigenvector: List[complex]


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
    def Nkx(self) -> int:
        return 2 * self.resolution[0] + 1

    @property
    def nkx_points(self):
        # return np.fft.fftfreq(self.resolution[0])
        return np.arange(-self.resolution[0], self.resolution[0] + 1, dtype=int)

    @property
    def Nky(self) -> int:
        return 2 * self.resolution[1] + 1

    @property
    def nky_points(self):
        return np.arange(-self.resolution[1], self.resolution[1] + 1, dtype=int)

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
        xt, yt, zt = np.meshgrid(
            self.nkx_points,
            self.nky_points,
            self.nz_points,
            indexing="ij",
        )
        return np.array([xt.ravel(), yt.ravel(), zt.ravel()]).T

    def get_index(self, nkx: int, nky: int, nz: int) -> int:
        ikx = (nkx + self.resolution[0]) * self.Nky * self.Nkz
        iky = (nky + self.resolution[1]) * self.Nkz
        return ikx + iky + nz

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
            (nkx1, nkx2, nz) = coordinates[arg]
            e = eigenvector_array[arg]
            x_phase = (nkx1 * self.dkx0[0] + nkx2 * self.dkx1[0] + kx) * points[:, 0]
            y_phase = (nkx1 * self.dkx0[1] + nkx2 * self.dkx1[1] + ky) * points[:, 1]
            out += (
                e
                * calculate_sho_wavefunction(
                    points[:, 2], self.sho_omega, self.mass, nz
                )
                * np.exp(1j * (x_phase + y_phase))
            )
        return out

    def calculate_wavefunction_fast(
        self,
        eigenstate: Eigenstate,
        points: ArrayLike,
    ) -> NDArray:
        return calculate_wavefunction_fast(self._config, eigenstate, points)
