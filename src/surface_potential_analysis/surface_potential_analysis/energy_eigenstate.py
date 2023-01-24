import json
from functools import cached_property
from pathlib import Path
from typing import List, Tuple, TypedDict

import numpy as np
import scipy
from numpy.typing import ArrayLike, NDArray

import hamiltonian_generator

from .energy_data import EnergyInterpolation
from .sho_wavefunction import calculate_sho_wavefunction


class EigenstateConfig(TypedDict):
    resolution: Tuple[int, int, int]
    """Resolution in x,y,z to produce the eigenstates in"""
    sho_omega: float
    """Angular frequency (in rad s-1) of the sho we will fit using"""
    mass: float
    """Mass in Kg"""
    delta_x1: Tuple[float, float]
    """maximum extent in the x direction"""
    delta_x2: Tuple[float, float]
    """maximum extent in the x direction"""


class EigenstateConfigLegacy(TypedDict):
    resolution: Tuple[int, int, int]
    sho_omega: float
    mass: float
    delta_x: float
    delta_y: float


class Eigenstate(TypedDict):
    kx: float
    ky: float
    eigenvector: List[complex]


class EnergyEigenstates(TypedDict):
    eigenstate_config: EigenstateConfig
    kx_points: List[float]
    ky_points: List[float]
    eigenvalues: List[float]
    eigenvectors: List[List[complex]]


class EnergyEigenstatesRawLegacy(TypedDict):
    eigenstate_config: EigenstateConfigLegacy
    kx_points: List[float]
    ky_points: List[float]
    eigenvalues: List[float]
    eigenvectors_re: List[List[float]]
    eigenvectors_im: List[List[float]]


class EnergyEigenstatesRaw(TypedDict):
    eigenstate_config: EigenstateConfig
    kx_points: List[float]
    ky_points: List[float]
    eigenvalues: List[float]
    eigenvectors_re: List[List[float]]
    eigenvectors_im: List[List[float]]


def config_from_legacy(config: EigenstateConfigLegacy) -> EigenstateConfig:
    return {
        "resolution": config["resolution"],
        "delta_x1": (config["delta_x"], 0),
        "delta_x2": (0, config["delta_y"]),
        "mass": config["mass"],
        "sho_omega": config["sho_omega"],
    }


def save_energy_eigenstates(data: EnergyEigenstates, path: Path) -> None:
    with path.open("w") as f:
        out: EnergyEigenstatesRaw = {
            "eigenstate_config": data["eigenstate_config"],
            "eigenvalues": data["eigenvalues"],
            "eigenvectors_re": np.real(data["eigenvectors"]).tolist(),
            "eigenvectors_im": np.imag(data["eigenvectors"]).tolist(),
            "kx_points": data["kx_points"],
            "ky_points": data["ky_points"],
        }
        json.dump(out, f)


def load_energy_eigenstates_old(path: Path) -> EnergyEigenstates:
    with path.open("r") as f:
        out: EnergyEigenstatesRawLegacy = json.load(f)

        eigenvectors = np.array(out["eigenvectors_re"]) + 1j * np.array(
            out["eigenvectors_im"]
        )

        return {
            "eigenstate_config": config_from_legacy(out["eigenstate_config"]),
            "eigenvalues": out["eigenvalues"],
            "eigenvectors": eigenvectors.tolist(),
            "kx_points": out["kx_points"],
            "ky_points": out["ky_points"],
        }


def append_energy_eigenstates(
    path: Path, eigenstate: Eigenstate, eigenvalue: float
) -> None:
    with path.open("r") as f:
        data: EnergyEigenstatesRawLegacy = json.load(f)
        data["kx_points"].append(eigenstate["kx"])
        data["ky_points"].append(eigenstate["ky"])
        data["eigenvalues"].append(eigenvalue)
        data["eigenvectors_re"].append(np.real(eigenstate["eigenvector"]).tolist())
        data["eigenvectors_im"].append(np.imag(eigenstate["eigenvector"]).tolist())

    with path.open("w") as f:
        json.dump(data, f)


def get_eigenstate_list(eigenstates: EnergyEigenstates) -> List[Eigenstate]:
    return [
        {
            "eigenvector": eigenvector,
            "kx": eigenstates["kx_points"][i],
            "ky": eigenstates["ky_points"][i],
        }
        for (i, eigenvector) in enumerate(eigenstates["eigenvectors"])
    ]


def calculate_wavefunction_fast(
    config: EigenstateConfig, eigenstate: Eigenstate, points: ArrayLike
) -> NDArray:
    assert config["delta_x1"][1] == 0
    assert config["delta_x2"][0] == 0
    return np.array(
        hamiltonian_generator.get_eigenstate_wavefunction(
            config["resolution"],
            config["delta_x1"][0],
            config["delta_x2"][1],
            config["mass"],
            config["sho_omega"],
            eigenstate["kx"],
            eigenstate["ky"],
            eigenstate["eigenvector"],
            np.array(points).tolist(),
        ),
        dtype=complex,
    )


class EigenstateConfigUtil:

    _config: EigenstateConfig

    def __init__(self, config: EigenstateConfig) -> None:
        self._config = config

    @property
    def resolution(self):
        return self._config["resolution"]

    @property
    def mass(self):
        return self._config["mass"]

    @property
    def sho_omega(self):
        return self._config["sho_omega"]

    @cached_property
    def _dk_prefactor(self):
        # See https://physics.stackexchange.com/questions/340860/reciprocal-lattice-in-2d
        x1_part = self.delta_x1[0] * self.delta_x2[1]
        x2_part = self.delta_x1[1] * self.delta_x2[0]
        return (2 * np.pi) / (x1_part - x2_part)

    @property
    def delta_x1(self) -> Tuple[float, float]:
        return self._config["delta_x1"]

    @cached_property
    def dkx1(self) -> Tuple[float, float]:
        return (
            self._dk_prefactor * self.delta_x2[1],
            -self._dk_prefactor * self.delta_x2[0],
        )

    @property
    def Nkx(self) -> int:
        return 2 * self.resolution[0] + 1

    @property
    def nkx_points(self):
        return np.arange(-self.resolution[0], self.resolution[0] + 1, dtype=int)

    @property
    def delta_x2(self) -> Tuple[float, float]:
        return self._config["delta_x2"]

    @cached_property
    def dkx2(self) -> Tuple[float, float]:
        return (
            -self._dk_prefactor * self.delta_x1[1],
            self._dk_prefactor * self.delta_x1[0],
        )

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

    @cached_property
    def coordinates(self) -> NDArray:
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
        coordinates = self.coordinates
        args = (
            np.arange(self.coordinates.shape[0])
            if cutoff is None
            else np.argsort(np.abs(eigenvector_array))[::-1][:cutoff]
        )
        kx = eigenstate["kx"]
        ky = eigenstate["ky"]
        for arg in args:
            (nkx1, nkx2, nz) = coordinates[arg]
            e = eigenvector_array[arg]
            x_phase = (nkx1 * self.dkx1[0] + nkx2 * self.dkx2[0] + kx) * points[:, 0]
            y_phase = (nkx1 * self.dkx1[1] + nkx2 * self.dkx2[1] + ky) * points[:, 1]
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


def get_minimum_coordinate(arr: ArrayLike) -> Tuple[int, ...]:
    points = np.array(arr)
    return np.unravel_index(np.argmin(points), points.shape)  # type: ignore


def generate_sho_config_minimum(
    interpolation: EnergyInterpolation,
    mass: float,
    *,
    initial_guess: float = 1e14,
    fit_max_energy_fraction=0.5
) -> Tuple[float, float]:
    points = np.array(interpolation["points"])
    min_coord = get_minimum_coordinate(points)
    min_z = min_coord[2]
    z_points = points[min_coord[0], min_coord[1]]
    z_indexes = np.arange(z_points.shape[0])

    far_edge_energy = z_points[-1]
    # We choose a region that is suitably harmonic
    # ie we cut off the tail of the potential
    fit_max_energy = fit_max_energy_fraction * far_edge_energy
    above_threshold = (z_indexes > min_z) & (z_points > fit_max_energy)
    # Stops at the first above threshold
    max_index: int = int(np.argmax(above_threshold) - 1)
    above_threshold = (z_indexes < min_z) & (z_points > fit_max_energy)

    # Search backwards, stops at the first above threshold
    min_index: int = z_points.shape[0] - np.argmax(above_threshold[::-1])

    print("min", min_index, "middle", min_z, "max", max_index)
    z_offset = -interpolation["dz"] * min_z
    # Fit E = 1/2 * m * sho_omega ** 2 * z**2
    def fitting_f(z, sho_omega):
        return 0.5 * mass * (sho_omega * z) ** 2

    opt_params, _cov = scipy.optimize.curve_fit(
        f=fitting_f,
        xdata=np.arange(min_index, max_index + 1) * interpolation["dz"] + z_offset,
        ydata=z_points[min_index : max_index + 1],
        p0=[initial_guess],
    )

    return opt_params[0], z_offset


def get_bloch_phases(data: EnergyEigenstates, origin_point: Tuple[float, float, float]):
    util = EigenstateConfigUtil(data["eigenstate_config"])

    phases: List[float] = []
    for eigenstate in get_eigenstate_list(data):
        point_at_origin = util.calculate_wavefunction_fast(eigenstate, [origin_point])
        phases.append(float(np.angle(point_at_origin[0])))
    return phases


def normalize_eigenstate_phase(
    data: EnergyEigenstates, origin_point: Tuple[float, float, float]
) -> EnergyEigenstates:

    eigenvectors = data["eigenvectors"]

    phases = get_bloch_phases(data, origin_point=origin_point)
    phase_factor = np.real_if_close(np.exp(-1j * np.array(phases)))
    fixed_phase_eigenvectors = np.multiply(eigenvectors, phase_factor[:, np.newaxis])

    return {
        "eigenvalues": data["eigenvalues"],
        "eigenvectors": fixed_phase_eigenvectors.tolist(),
        "kx_points": data["kx_points"],
        "ky_points": data["ky_points"],
        "eigenstate_config": data["eigenstate_config"],
    }


def filter_eigenstates_grid(
    eigenstates: EnergyEigenstates, kx_points: List[float], ky_points: List[float]
) -> EnergyEigenstates:
    filtered_kx = np.zeros_like(eigenstates["kx_points"], dtype=bool)
    for kx in kx_points:
        filtered_kx = np.logical_or(filtered_kx, np.equal(eigenstates["kx_points"], kx))

    filtered_ky = np.zeros_like(eigenstates["kx_points"], dtype=bool)
    for ky in ky_points:
        filtered_ky = np.logical_or(filtered_ky, np.equal(eigenstates["ky_points"], ky))

    filtered = np.logical_and(filtered_kx, filtered_ky)

    return {
        "eigenstate_config": eigenstates["eigenstate_config"],
        "eigenvalues": np.array(eigenstates["eigenvalues"])[filtered].tolist(),
        "eigenvectors": np.array(eigenstates["eigenvectors"])[filtered].tolist(),
        "kx_points": np.array(eigenstates["kx_points"])[filtered].tolist(),
        "ky_points": np.array(eigenstates["ky_points"])[filtered].tolist(),
    }


def filter_eigenstates_n_point(eigenstates: EnergyEigenstates, n: int):
    """Given Energy Eigenstates produce a reduced grid spacing"""
    # For an 8 point grid we have eigenstate_len of 16
    eigenstate_len = np.unique(eigenstates["kx_points"]).shape[0]
    # We want an 8x8 grid of eigenstates, so for an 8 point grid we take every 2 points
    take_every = eigenstate_len // (2 * n)
    if take_every == 0:
        raise Exception("Not enough k-points in the grid")

    kx_points = np.sort(np.unique(eigenstates["kx_points"]))[0::take_every].tolist()
    ky_points = np.sort(np.unique(eigenstates["ky_points"]))[0::take_every].tolist()
    return filter_eigenstates_grid(eigenstates, kx_points, ky_points)
