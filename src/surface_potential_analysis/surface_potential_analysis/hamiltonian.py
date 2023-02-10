import datetime
import os
from functools import cache, wraps
from pathlib import Path
from typing import Any, Callable, List, Tuple, TypeVar

import numpy as np
from numpy.typing import NDArray
from scipy.constants import hbar

import hamiltonian_generator
from surface_potential_analysis.brillouin_zone import grid_space

from .energy_data import EnergyInterpolation
from .energy_eigenstate import (
    Eigenstate,
    EigenstateConfig,
    EigenstateConfigUtil,
    EnergyEigenstates,
    append_energy_eigenstates,
    get_brillouin_points_irreducible_config,
    save_energy_eigenstates,
)
from .sho_wavefunction import calculate_sho_wavefunction

F = TypeVar("F", bound=Callable)


def timed(f: F) -> F:
    @wraps(f)
    def wrap(*args, **kw):
        ts = datetime.datetime.now()
        result = f(*args, **kw)
        te = datetime.datetime.now()
        print(f"func: {f.__name__} took: {(te - ts).total_seconds()} sec")
        return result

    return wrap  # type: ignore


class SurfaceHamiltonianUtil(EigenstateConfigUtil):

    _potential: EnergyInterpolation

    _potential_offset: float

    def __init__(
        self,
        config: EigenstateConfig,
        potential: EnergyInterpolation,
        potential_offset: float,
    ) -> None:
        super().__init__(config)
        self._potential = potential
        self._potential_offset = potential_offset

        if self.Nkx != self.Nx:
            raise AssertionError("Potential has the incorrect shape")

        if (2 * self.Nkx) > self.Nx:
            print("Warning: max(ndkx) > Nx, some over sampling will occur")
        if (2 * self.Nky) > self.Ny:
            print("Warning: max(ndky) > Ny, some over sampling will occur")

    @property
    def points(self):
        return np.atleast_3d(self._potential["points"])

    @property
    def z_offset(self):
        return self._potential_offset

    @property
    def Nx(self) -> int:
        return self.points.shape[0]

    @property
    def Ny(self) -> int:
        return self.points.shape[1]

    @property
    def lattuice_coordinates(self) -> NDArray:
        """
        Lattice coordinates as calculated from delta_x0, delta_x1 with the origin at the center
        Note we dont include the repeated symmetry point in the potential
        """
        return grid_space(
            self.delta_x0, self.delta_x1, shape=(self.Nx, self.Ny), endpoint=False
        )

    @property
    def dz(self) -> float:
        return self._potential["dz"]

    @property
    def Nz(self) -> int:
        return self.points.shape[2]

    @property
    def z_points(self):
        z_start = self.z_offset
        z_end = self.z_offset + (self.Nz - 1) * self.dz
        return np.linspace(z_start, z_end, self.Nz, dtype=float)

    def hamiltonian(self, kx: float, ky: float) -> NDArray:
        diagonal_energies = np.diag(self._calculate_diagonal_energy(kx, ky))
        other_energies = self._calculate_off_diagonal_energies_fast()

        energies = diagonal_energies + other_energies
        if os.environ.get("DEBUG_CHECKS", False) and not np.allclose(
            energies, energies.conjugate().T
        ):
            raise AssertionError("Hamiltonian is not hermitian")
        return energies

    @timed
    def _calculate_diagonal_energy(self, kx: float, ky: float) -> NDArray[Any]:
        kx1_coords, kx2_coords, nz_coords = self.eigenstate_indexes.T

        kx_points = self.dkx0[0] * kx1_coords + self.dkx1[0] * kx2_coords + kx
        x_energy = (hbar * kx_points) ** 2 / (2 * self.mass)
        ky_points = self.dkx0[1] * kx1_coords + self.dkx1[1] * kx2_coords + ky
        y_energy = (hbar * ky_points) ** 2 / (2 * self.mass)
        z_energy = (hbar * self.sho_omega) * (nz_coords + 0.5)
        return x_energy + y_energy + z_energy

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

        if os.environ.get("DEBUG_CHECKS", False) and not np.all(
            np.isreal(np.real_if_close(fft_potential))
        ):
            raise AssertionError("FFT was not real!")
        return np.real_if_close(fft_potential)

    @cache
    def _calculate_sho_wavefunction_points(self, n: int) -> NDArray:
        """Generates the nth SHO wavefunction using the current config"""
        return calculate_sho_wavefunction(self.z_points, self.sho_omega, self.mass, n)

    @cache
    @timed
    def _calculate_off_diagonal_energies_fast(self) -> NDArray:

        return np.array(
            hamiltonian_generator.calculate_off_diagonal_energies(
                self.get_ft_potential().tolist(),
                self.resolution,
                self.dz,
                self.mass,
                self.sho_omega,
                self.z_offset,
            )
        )

    @cache
    def _calculate_off_diagonal_entry(self, nz1, nz2, ndkx, ndky) -> float:
        """Calculates the off diagonal energy using the 'folded' points ndkx, ndky"""
        ft_pot_points = self.get_ft_potential()[ndkx, ndky]
        hermite1 = self._calculate_sho_wavefunction_points(nz1)
        hermite2 = self._calculate_sho_wavefunction_points(nz2)

        fourier_transform = float(np.sum(hermite1 * hermite2 * ft_pot_points))

        return self.dz * fourier_transform

    def _calculate_off_diagonal_energies(self) -> NDArray:

        n_coordinates = len(self.eigenstate_indexes)
        hamiltonian = np.zeros(shape=(n_coordinates, n_coordinates))

        for (index1, [nkx1, nky1, nz1]) in enumerate(self.eigenstate_indexes):
            for (index2, [nkx2, nky2, nz2]) in enumerate(self.eigenstate_indexes):
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

    @timed
    def calculate_eigenvalues(self, kx, ky) -> Tuple[List[float], List[Eigenstate]]:
        """
        Returns the eigenvalues as a list of vectors,
        ie v[i] is the eigenvector associated to the eigenvalue w[i]
        """
        w, v = np.linalg.eigh(self.hamiltonian(kx, ky))
        return (w.tolist(), [{"eigenvector": vec, "kx": kx, "ky": ky} for vec in v.T])


def generate_energy_eigenstates_grid(
    hamiltonian: SurfaceHamiltonianUtil,
    k_points: NDArray,
    path: Path,
    *,
    include_bands: List[int] | None = None,
) -> None:
    include_bands = [0] if include_bands is None else include_bands
    data: EnergyEigenstates = {
        "kx_points": [],
        "ky_points": [],
        "eigenvalues": [],
        "eigenvectors": [],
        "eigenstate_config": hamiltonian._config,
    }
    save_energy_eigenstates(data, path)

    for (kx, ky) in k_points:
        e_vals, e_states = hamiltonian.calculate_eigenvalues(kx, ky)
        a_min = np.argpartition(e_vals, include_bands)

        for idx in include_bands:
            eigenvalue = e_vals[a_min[idx]]
            eigenstate = e_states[a_min[idx]]
            append_energy_eigenstates(path, eigenstate, eigenvalue)


def generate_energy_eigenstates_grid_copper_100(
    path: Path,
    hamiltonian: SurfaceHamiltonianUtil,
    *,
    size=(8, 8),
    include_zero=True,
    include_bands: List[int] | None = None,
):
    k_points = get_brillouin_points_irreducible_config(
        hamiltonian._config, size=size, include_zero=include_zero
    )

    return generate_energy_eigenstates_grid(
        hamiltonian, k_points, path, include_bands=include_bands
    )


def calculate_energy_eigenstates(
    hamiltonian: SurfaceHamiltonianUtil,
    kx_points: NDArray,
    ky_points: NDArray,
    *,
    include_bands: List[int] | None = None,
) -> EnergyEigenstates:

    include_bands = [0] if include_bands is None else include_bands
    out: EnergyEigenstates = {
        "eigenstate_config": hamiltonian._config,
        "kx_points": [],
        "ky_points": [],
        "eigenvalues": [],
        "eigenvectors": [],
    }

    for (kx, ky) in zip(kx_points, ky_points):
        e_vals, e_states = hamiltonian.calculate_eigenvalues(kx, ky)
        a_min = np.argpartition(e_vals, include_bands)

        for idx in include_bands:
            eigenvalue = e_vals[a_min[idx]]
            eigenstate = e_states[a_min[idx]]

            out["kx_points"].append(eigenstate["kx"])
            out["ky_points"].append(eigenstate["ky"])
            out["eigenvalues"].append(eigenvalue)
            out["eigenvectors"].append(eigenstate["eigenvector"])

    return out
