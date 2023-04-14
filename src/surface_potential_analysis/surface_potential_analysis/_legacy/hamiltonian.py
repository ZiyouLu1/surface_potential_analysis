from functools import cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.constants import hbar

import hamiltonian_generator
from surface_potential_analysis._legacy.brillouin_zone import grid_space

from ..util import timed
from .energy_data import EnergyInterpolation
from .energy_eigenstate import (
    Eigenstate,
    EigenstateConfig,
    EigenstateConfigUtil,
    EnergyEigenstatesLegacy,
    append_energy_eigenstates,
    get_brillouin_points_irreducible_config,
    save_energy_eigenstates,
)
from .sho_wavefunction import calculate_sho_wavefunction


def diag_along_axis(points: NDArray, axis=-1):
    return np.apply_along_axis(np.diag, axis, points)


def get_hamiltonian_from_potential(potential: NDArray):
    """
    Given a potential in position basis [ix0, ix1, ix2]
    Gets the hamiltonian of a potential for a given points, in position basis,
    with indexing [ix0, ix1, ix2, jx0, jx1, jx2].
    This is just a matrix with the potential along the diagonals
    """
    hamiltonian = diag_along_axis(
        diag_along_axis(diag_along_axis(potential, axis=2), axis=1),
        axis=0,
    )
    hamiltonian = np.moveaxis(hamiltonian, 1, -1)
    hamiltonian = np.moveaxis(hamiltonian, 2, -1)
    hamiltonian = np.moveaxis(hamiltonian, 3, -1)

    return hamiltonian


def transform_hamiltonian_into_momentum_basis(hamiltonian: NDArray):
    """
    Given a hamiltonian in position basis, with indexing [ix0, ix1, ix2, jx0, jx1, jx2]
    return the hamiltonian in momentum basis
    """
    return np.fft.ifftn(
        np.fft.fftn(hamiltonian, axes=(0, 1, 2), norm="ortho"),
        axes=(3, 4, 5),
        norm="ortho",
    )


def transform_hamiltonian_into_position_basis(hamiltonian: NDArray):
    """
    Given a hamiltonian in momentum basis, with indexing [ikx0, ikx1, ikx2, jkx0, jkx1, jkx2]
    return the hamiltonian in position basis
    """
    return np.fft.fftn(
        np.fft.ifftn(hamiltonian, axes=(0, 1, 2), norm="ortho"),
        axes=(3, 4, 5),
        norm="ortho",
    )


def flatten_hamiltonian(hamiltonian: NDArray):
    """
    Given a hamiltonian with indexing [ix0, ix1, ix2, jx0, jx1, jx2]
    return the flattenned hamiltonian with indexing [i,j]
    """
    print(hamiltonian.shape, hamiltonian.shape[0:3])
    n_states = np.prod(hamiltonian.shape[0:3])

    # x0t, x1t, x2t = np.meshgrid(
    #     np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
    # )
    # coords = np.array([x0t.ravel(), x1t.ravel(), x2t.ravel()]).T
    # out = np.zeros((n_states,n_states))

    return hamiltonian.reshape(n_states, n_states)


def unflatten_hamiltonian(hamiltonian: NDArray, shape: Sequence[int]):
    return hamiltonian.reshape((*shape, *shape))


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
        if 2 * (self.Nkx0 - 1) > self.Nx:
            print(self.Nkx0, self.Nx)
            raise AssertionError(
                "Potential does not have enough resolution in x direction"
            )

        if 2 * (self.Nkx1 - 1) > self.Ny:
            print(self.Nkx1, self.Ny)
            raise AssertionError(
                "Potential does not have enough resolution in y direction"
            )

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

    @timed
    def hamiltonian(self, kx: float, ky: float) -> NDArray:
        diagonal_energies = np.diag(self._calculate_diagonal_energy(kx, ky))
        other_energies = self._calculate_off_diagonal_energies_fast()

        energies = diagonal_energies + other_energies

        if False and not np.allclose(energies, energies.conjugate().T):
            raise AssertionError("Hamiltonian is not hermitian")

        return energies

    @timed
    def _calculate_diagonal_energy(self, kx: float, ky: float) -> NDArray[Any]:
        kx0_coords, kx1_coords, nkz_coords = self.eigenstate_indexes.T

        kx_points = self.dkx0[0] * kx0_coords + self.dkx1[0] * kx1_coords + kx
        x_energy = (hbar * kx_points) ** 2 / (2 * self.mass)
        ky_points = self.dkx0[1] * kx0_coords + self.dkx1[1] * kx1_coords + ky
        y_energy = (hbar * ky_points) ** 2 / (2 * self.mass)
        z_energy = (hbar * self.sho_omega) * (nkz_coords + 0.5)
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
        return np.fft.ifft2(subtracted_potential, axes=(0, 1))

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
    def _calculate_off_diagonal_entry(self, nz1, nz2, ndkx0, ndkx1) -> float:
        """Calculates the off diagonal energy using the 'folded' points ndkx, ndky"""
        ft_pot_points = self.get_ft_potential()[ndkx0, ndkx1]
        hermite1 = self._calculate_sho_wavefunction_points(nz1)
        hermite2 = self._calculate_sho_wavefunction_points(nz2)

        fourier_transform = float(np.sum(hermite1 * hermite2 * ft_pot_points))

        return self.dz * fourier_transform

    def _calculate_off_diagonal_energies(self) -> NDArray:
        n_coordinates = len(self.eigenstate_indexes)
        hamiltonian = np.zeros(shape=(n_coordinates, n_coordinates))

        for index1, [nkx0_0, nkx1_0, nz1] in enumerate(self.eigenstate_indexes):
            for index2, [nkx0_1, nkx1_1, nz2] in enumerate(self.eigenstate_indexes):
                # Number of jumps in units of dkx for this matrix element

                # As k-> k+ Nx * dkx the ft potential is left unchanged
                # Therefore we 'wrap round' ndkx into the region 0<ndkx<Nx
                # Where Nx is the number of x points

                # In reality we want to make sure ndkx < Nx (ie we should
                # make sure we generate enough points in the interpolation)
                ndkx0 = (nkx0_1 - nkx0_0) % self.Nx
                ndkx1 = (nkx1_1 - nkx1_0) % self.Ny

                hamiltonian[index1, index2] = self._calculate_off_diagonal_entry(
                    nz1, nz2, ndkx0, ndkx1
                )

        return hamiltonian

    def _calculate_off_diagonal_energies_fourier(self) -> NDArray:
        hamiltonian = get_hamiltonian_from_potential(self.get_sho_subtracted_points())
        transformed = transform_hamiltonian_into_momentum_basis(hamiltonian)
        return flatten_hamiltonian(transformed)

    @timed
    def calculate_eigenvalues(
        self, kx: float, ky: float
    ) -> tuple[list[float], list[Eigenstate]]:
        """
        Returns the eigenvalues as a list of vectors,
        ie v[i] is the eigenvector associated to the eigenvalue w[i]
        """
        hamiltonian = self.hamiltonian(kx, ky)

        is_symmetric_x = np.array_equal(self.points[1:, :], self.points[:0:-1, :])
        is_symmetric_y = np.array_equal(self.points[:, 1:], self.points[:, :0:-1])
        # If the potential is symmetric the fourier transform is real
        # This provides us with a significant speedup
        if is_symmetric_x and is_symmetric_y:
            print("Is symmetric yay!")
            hamiltonian = np.real_if_close(hamiltonian)

        w, v = np.linalg.eigh(hamiltonian)

        return (w.tolist(), [{"eigenvector": vec, "kx": kx, "ky": ky} for vec in v.T])


def generate_energy_eigenstates_from_k_points(
    hamiltonian: SurfaceHamiltonianUtil,
    k_points: NDArray,
    *,
    save_bands: dict[int, Path],
) -> None:
    input("Warning: this might overwrite previous data. Press enter to continue...")
    for path in save_bands.values():
        data: EnergyEigenstatesLegacy = {
            "kx_points": [],
            "ky_points": [],
            "eigenvalues": [],
            "eigenvectors": [],
            "eigenstate_config": hamiltonian._config,
        }
        save_energy_eigenstates(data, path)

    for kx, ky in k_points:
        e_vals, e_states = hamiltonian.calculate_eigenvalues(kx, ky)
        a_min = np.argpartition(e_vals, list(save_bands.keys()))

        for index, path in save_bands.items():
            eigenvalue = e_vals[a_min[index]]
            eigenstate = e_states[a_min[index]]
            append_energy_eigenstates(path, eigenstate, eigenvalue)


def generate_energy_eigenstates_grid(
    hamiltonian: SurfaceHamiltonianUtil,
    *,
    size: tuple[int, int] = (8, 8),
    include_zero=True,
    save_bands: dict[int, Path],
):
    k_points = get_brillouin_points_irreducible_config(
        hamiltonian._config, size=size, include_zero=include_zero
    )

    return generate_energy_eigenstates_from_k_points(
        hamiltonian, k_points, save_bands=save_bands
    )


def calculate_energy_eigenstates(
    hamiltonian: SurfaceHamiltonianUtil,
    kx_points: NDArray,
    ky_points: NDArray,
    *,
    include_bands: list[int] | None = None,
) -> EnergyEigenstatesLegacy:
    include_bands = [0] if include_bands is None else include_bands
    out: EnergyEigenstatesLegacy = {
        "eigenstate_config": hamiltonian._config,
        "kx_points": [],
        "ky_points": [],
        "eigenvalues": [],
        "eigenvectors": [],
    }

    for kx, ky in zip(kx_points, ky_points):
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