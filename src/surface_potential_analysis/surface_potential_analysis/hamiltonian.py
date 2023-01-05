import datetime
import math
import os
from functools import cache, cached_property, wraps
from pathlib import Path
from typing import Any, Callable, List, Tuple, TypeVar

import numpy as np
import scipy.special
from numpy.typing import ArrayLike, NDArray
from scipy.constants import hbar

import hamiltonian_generator

from .energy_data.energy_data import EnergyInterpolation
from .energy_data.energy_eigenstates import (
    Eigenstate,
    EnergyEigenstates,
    WavepacketGrid,
    append_energy_eigenstates,
    get_eigenstate_list,
    save_energy_eigenstates,
)
from .energy_data.sho_config import EigenstateConfig

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


def calculate_sho_wavefunction(z_points, sho_omega, mass, n) -> NDArray:
    norm = (sho_omega * mass / hbar) ** 0.5
    normalized_z = z_points * norm

    prefactor = math.sqrt((norm / (2**n)) / (math.factorial(n) * math.sqrt(math.pi)))
    hermite = scipy.special.eval_hermite(n, normalized_z)
    exponential = np.exp(-np.square(normalized_z) / 2)
    return prefactor * hermite * exponential


class SurfaceHamiltonian:

    _potential: EnergyInterpolation

    _potential_offset: float

    _config: EigenstateConfig

    _resolution: Tuple[int, int, int]

    def __init__(
        self,
        resolution: Tuple[int, int, int],
        potential: EnergyInterpolation,
        config: EigenstateConfig,
        potential_offset: float,
    ) -> None:

        self._potential = potential
        self._potential_offset = potential_offset
        self._config = config
        self._resolution = resolution

        if (2 * self.Nkx) > self.Nx:
            print("Warning: max(ndkx) > Nx, some over sampling will occur")
        if (2 * self.Nky) > self.Ny:
            print("Warning: max(ndky) > Ny, some over sampling will occur")

    @property
    def points(self):
        return np.atleast_3d(self._potential["points"])

    @property
    def mass(self):
        return self._config["mass"]

    @property
    def sho_omega(self):
        return self._config["sho_omega"]

    @property
    def z_offset(self):
        return self._potential_offset

    @property
    def delta_x(self) -> float:
        return self._config["delta_x"]

    @cached_property
    def dkx(self) -> float:
        return 2 * np.pi / self.delta_x

    @property
    def Nx(self) -> int:
        return self.points.shape[0]

    @property
    def Nkx(self) -> int:
        return 2 * self._resolution[0] + 1

    @property
    def x_points(self):
        """
        Calculate the lattice coordinates in the x direction

        Note: We don't store the 'nth' pixel
        """
        return np.linspace(0, self.delta_x, self.Nx, endpoint=False)

    @property
    def nkx_points(self):
        return np.arange(-self._resolution[0], self._resolution[0] + 1, dtype=int)

    @property
    def delta_y(self) -> float:
        return self._config["delta_y"]

    @cached_property
    def dky(self) -> float:
        return 2 * np.pi / (self.delta_y)

    @property
    def Ny(self) -> int:
        return self.points.shape[1]

    @property
    def Nky(self) -> int:
        return 2 * self._resolution[1] + 1

    @property
    def y_points(self):
        """
        Calculate the lattice coordinates in the y direction

        Note: We don't store the 'nth' pixel
        """
        return np.linspace(0, self.delta_y, self.Ny, endpoint=False)

    @property
    def nky_points(self):
        return np.arange(-self._resolution[1], self._resolution[1] + 1, dtype=int)

    @property
    def dz(self) -> float:
        return self._potential["dz"]

    @property
    def Nz(self) -> int:
        return self.points.shape[2]

    @property
    def Nkz(self) -> int:
        return self._resolution[2]

    @property
    def z_points(self):
        z_start = self.z_offset
        z_end = self.z_offset + (self.Nz - 1) * self.dz
        return np.linspace(z_start, z_end, self.Nz, dtype=int)

    @property
    def nz_points(self):
        return np.arange(self.Nkz, dtype=int)

    @cached_property
    def coordinates(self) -> NDArray:
        xt, yt, zt = np.meshgrid(
            self.nkx_points,
            self.nky_points,
            self.nz_points,
            indexing="ij",
        )
        return np.array([xt.ravel(), yt.ravel(), zt.ravel()]).T

    def hamiltonian(self, kx: float, ky: float) -> NDArray:
        diagonal_energies = np.diag(self._calculate_diagonal_energy(kx, ky))
        other_energies = self._calculate_off_diagonal_energies_fast()

        energies = diagonal_energies + other_energies
        if os.environ.get("DEBUG_CHECKS", False) and not np.allclose(
            energies, energies.conjugate().T
        ):
            raise AssertionError("hamiltonian is not hermitian")
        return energies

    @timed
    def _calculate_diagonal_energy(self, kx: float, ky: float) -> NDArray[Any]:
        kx_coords, ky_coords, nz_coords = self.coordinates.T

        x_energy = (hbar * (self.dkx * kx_coords + kx)) ** 2 / (2 * self.mass)
        y_energy = (hbar * (self.dky * ky_coords + ky)) ** 2 / (2 * self.mass)
        z_energy = (hbar * self.sho_omega) * (nz_coords + 0.5)
        return x_energy + y_energy + z_energy

    def get_index(self, nkx: int, nky: int, nz: int) -> int:
        ikx = (nkx + self._resolution[0]) * self.Nky * self.Nkz
        iky = (nky + self._resolution[1]) * self.Nkz
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
    def _calculate_off_diagonal_entry(self, nz1, nz2, ndkx, ndky) -> float:
        """Calculates the off diagonal energy using the 'folded' points ndkx, ndky"""
        ft_pot_points = self.get_ft_potential()[ndkx, ndky]
        hermite1 = self._calculate_sho_wavefunction_points(nz1)
        hermite2 = self._calculate_sho_wavefunction_points(nz2)

        fourier_transform = np.sum(hermite1 * hermite2 * ft_pot_points)

        return self.dz * fourier_transform

    @cache
    @timed
    def _calculate_off_diagonal_energies_fast(self) -> NDArray:

        return np.array(
            hamiltonian_generator.get_hamiltonian(
                self.get_ft_potential().tolist(),
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

    @timed
    def calculate_wavefunction_slow(
        self,
        points: ArrayLike,
        eigenstate: Eigenstate,
        cutoff: int | None = None,
    ) -> NDArray:
        points = np.array(points)
        out = np.zeros(shape=(points.shape[0]), dtype=complex)

        eigenvector_array = np.array(eigenstate["eigenvector"])
        coordinates = self.coordinates
        args = (
            np.arange(self.coordinates.shape[0])
            if cutoff is None
            else np.argsort(np.abs(eigenvector_array))[::-1][:cutoff]
        )
        kx = eigenstate["kx"]
        ky = eigenstate["ky"]
        for arg in args:
            (nkx, nky, nz) = coordinates[arg]
            e = eigenvector_array[arg]
            x_phase = (nkx * self.dkx + kx) * points[:, 0]
            y_phase = (nky * self.dky + ky) * points[:, 1]
            out += (
                e
                * calculate_sho_wavefunction(
                    points[:, 2], self.sho_omega, self.mass, nz
                )
                * np.exp(1j * (x_phase + y_phase))
            )
        return out

    @timed
    def calculate_wavefunction_fast(
        self,
        points: ArrayLike,
        eigenstate: Eigenstate,
    ) -> NDArray:
        return np.array(
            hamiltonian_generator.get_eigenstate_wavefunction(
                self._resolution,
                self.delta_x,
                self.delta_y,
                self.mass,
                self.sho_omega,
                eigenstate["kx"],
                eigenstate["ky"],
                eigenstate["eigenvector"],
                np.array(points).tolist(),
            )
        )


@timed
def calculate_eigenvalues(
    hamiltonian: SurfaceHamiltonian, kx: float, ky: float
) -> Tuple[NDArray, NDArray]:
    """
    Returns the eigenvalues as a list of vectors,
    ie v[i] is the eigenvector associated to the eigenvalue w[i]
    """
    w, v = np.linalg.eigh(hamiltonian.hamiltonian(kx, ky))
    return (w, v.T)


def calculate_wavepacket_grid_with_edge(
    eigenstates: EnergyEigenstates,
) -> WavepacketGrid:
    hamiltonian = SurfaceHamiltonian(
        resolution=eigenstates["resolution"],
        config=eigenstates["eigenstate_config"],
        potential={"dz": 0, "points": []},
        potential_offset=0,
    )

    x_points = np.linspace(-hamiltonian.delta_x, hamiltonian.delta_x / 2, 25)  # 49 97
    y_points = np.linspace(-hamiltonian.delta_y, hamiltonian.delta_y / 2, 25)
    z_points = np.linspace(-hamiltonian.delta_y, hamiltonian.delta_y, 21)

    xv, yv, zv = np.meshgrid(x_points, y_points, z_points)
    points = np.array([xv.ravel(), yv.ravel(), zv.ravel()]).T

    if not np.array_equal(xv, xv.ravel().reshape(xv.shape)):
        raise AssertionError("Error unraveling points")

    out = np.zeros_like(xv, dtype=complex)
    max_kx_point = np.max(eigenstates["kx_points"])
    max_ky_point = np.max(eigenstates["ky_points"])
    min_kx_point = np.min(eigenstates["kx_points"])
    min_ky_point = np.min(eigenstates["ky_points"])
    for eigenstate in get_eigenstate_list(eigenstates):
        print("pass")
        wfn = hamiltonian.calculate_wavefunction_fast(
            points,
            eigenstate,
        )

        is_kx_edge = (
            eigenstate["kx"] == max_kx_point or eigenstate["kx"] == min_kx_point
        )
        is_ky_edge = (
            eigenstate["ky"] == max_ky_point or eigenstate["ky"] == min_ky_point
        )
        edge_factor = (0.5 if is_kx_edge else 1.0) * (0.5 if is_ky_edge else 1.0)
        out += edge_factor * wfn.reshape(xv.shape) / len(eigenstates["eigenvectors"])

    return {
        "x_points": x_points.tolist(),
        "y_points": y_points.tolist(),
        "z_points": z_points.tolist(),
        "points": out.tolist(),
    }


def calculate_wavepacket_grid(
    eigenstates: EnergyEigenstates, cutoff: int | None = None
) -> WavepacketGrid:
    hamiltonian = SurfaceHamiltonian(
        resolution=eigenstates["resolution"],
        config=eigenstates["eigenstate_config"],
        potential={"dz": 0, "points": []},
        potential_offset=0,
    )

    x_points = np.linspace(-hamiltonian.delta_x, hamiltonian.delta_x / 2, 49)  # 97
    y_points = np.linspace(-hamiltonian.delta_y, hamiltonian.delta_y / 2, 49)
    z_points = np.linspace(-hamiltonian.delta_y, hamiltonian.delta_y, 21)

    xv, yv, zv = np.meshgrid(x_points, y_points, z_points)
    points = np.array([xv.ravel(), yv.ravel(), zv.ravel()]).T

    if not np.array_equal(xv, xv.ravel().reshape(xv.shape)):
        raise AssertionError("Error unraveling points")

    out = np.zeros_like(xv, dtype=complex)
    for eigenstate in get_eigenstate_list(eigenstates):
        print("pass")
        wfn = (
            hamiltonian.calculate_wavefunction_slow(
                points,
                eigenstate,
                cutoff=cutoff,
            )
            if cutoff is not None
            else hamiltonian.calculate_wavefunction_fast(points, eigenstate)
        )
        out += wfn.reshape(xv.shape) / len(eigenstates["eigenvectors"])

    return {
        "x_points": x_points.tolist(),
        "y_points": y_points.tolist(),
        "z_points": z_points.tolist(),
        "points": out.tolist(),
    }


def generate_energy_eigenstates_grid(
    path: Path, hamiltonian: SurfaceHamiltonian, grid_size=5, include_zero=True
) -> None:
    data: EnergyEigenstates = {
        "kx_points": [],
        "ky_points": [],
        "resolution": hamiltonian._resolution,
        "eigenvalues": [],
        "eigenvectors": [],
        "eigenstate_config": hamiltonian._config,
    }
    save_energy_eigenstates(data, path)

    dkx = hamiltonian.dkx
    (kx_points, kx_step) = np.linspace(
        -dkx / 2, dkx / 2, 2 * grid_size, endpoint=False, retstep=True
    )
    dky = hamiltonian.dky
    (ky_points, ky_step) = np.linspace(
        -dky / 2, dky / 2, 2 * grid_size, endpoint=False, retstep=True
    )
    if not include_zero:
        kx_points += kx_step / 2
        ky_points += ky_step / 2

    for kx in kx_points:
        for ky in ky_points:
            e_vals, e_vecs = calculate_eigenvalues(hamiltonian, kx, ky)
            a_min = np.argmin(e_vals)

            eigenvalue = e_vals[a_min]
            eigenvector = e_vecs[a_min].tolist()
            append_energy_eigenstates(
                path, {"eigenvector": eigenvector, "kx": kx, "ky": ky}, eigenvalue
            )


def calculate_energy_eigenstates(
    hamiltonian: SurfaceHamiltonian, kx_points: NDArray, ky_points: NDArray
) -> EnergyEigenstates:
    eigenvalues = []
    eigenvectors = []
    for (kx, ky) in zip(kx_points, ky_points):
        e_vals, e_vecs = calculate_eigenvalues(hamiltonian, kx, ky)
        a_min = np.argmin(e_vals)

        eigenvalues.append(e_vals[a_min])
        eigenvectors.append(e_vecs[a_min].tolist())

    return {
        "eigenstate_config": hamiltonian._config,
        "kx_points": kx_points.tolist(),
        "ky_points": ky_points.tolist(),
        "resolution": hamiltonian._resolution,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
    }


def get_wavepacket_phases(
    hamiltonian: SurfaceHamiltonian,
    data: EnergyEigenstates,
):
    h = hamiltonian
    origin_point = [h.delta_x / 2, h.delta_y / 2, 0]

    phases: List[float] = []
    for eigenstate in get_eigenstate_list(data):
        point_at_origin = hamiltonian.calculate_wavefunction_fast(
            [origin_point], eigenstate
        )
        phases.append(float(np.angle(point_at_origin[0])))
    return phases


def normalize_eigenstate_phase(
    hamiltonian: SurfaceHamiltonian, data: EnergyEigenstates
) -> EnergyEigenstates:

    eigenvectors = data["eigenvectors"]

    phases = get_wavepacket_phases(hamiltonian, data)
    phase_factor = np.real_if_close(np.exp(-1j * np.array(phases)))
    fixed_phase_eigenvectors = np.multiply(eigenvectors, phase_factor[:, np.newaxis])

    return {
        "eigenvalues": data["eigenvalues"],
        "eigenvectors": fixed_phase_eigenvectors.tolist(),
        "resolution": data["resolution"],
        "kx_points": data["kx_points"],
        "ky_points": data["ky_points"],
        "eigenstate_config": data["eigenstate_config"],
    }
