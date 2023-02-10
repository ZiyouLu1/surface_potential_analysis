import json
from pathlib import Path
from typing import List, Tuple, TypedDict

import numpy as np
import scipy
from numpy.typing import ArrayLike

from .brillouin_zone import get_points_in_brillouin_zone
from .eigenstate import Eigenstate, EigenstateConfig, EigenstateConfigUtil
from .energy_data import EnergyInterpolation


class EigenstateConfigRaw(TypedDict):
    resolution: List[int]
    sho_omega: float
    mass: float
    delta_x1: List[float]
    delta_x2: List[float]


class EnergyEigenstates(TypedDict):
    eigenstate_config: EigenstateConfig
    kx_points: List[float]
    ky_points: List[float]
    eigenvalues: List[float]
    eigenvectors: List[List[complex]]


class EnergyEigenstatesRaw(TypedDict):
    eigenstate_config: EigenstateConfigRaw
    kx_points: List[float]
    ky_points: List[float]
    eigenvalues: List[float]
    eigenvectors_re: List[List[float]]
    eigenvectors_im: List[List[float]]


def save_energy_eigenstates(data: EnergyEigenstates, path: Path) -> None:
    with path.open("w") as f:
        out: EnergyEigenstatesRaw = {
            "eigenstate_config": {
                "delta_x1": list(data["eigenstate_config"]["delta_x0"]),
                "delta_x2": list(data["eigenstate_config"]["delta_x1"]),
                "mass": data["eigenstate_config"]["mass"],
                "resolution": list(data["eigenstate_config"]["resolution"]),
                "sho_omega": data["eigenstate_config"]["sho_omega"],
            },
            "eigenvalues": data["eigenvalues"],
            "eigenvectors_re": np.real(data["eigenvectors"]).tolist(),
            "eigenvectors_im": np.imag(data["eigenvectors"]).tolist(),
            "kx_points": data["kx_points"],
            "ky_points": data["ky_points"],
        }
        json.dump(out, f)


def eigenstates_config_from_raw(raw: EigenstateConfigRaw) -> EigenstateConfig:
    return {
        "resolution": (
            raw["resolution"][0],
            raw["resolution"][1],
            raw["resolution"][2],
        ),
        "delta_x0": (raw["delta_x1"][0], raw["delta_x1"][1]),
        "delta_x1": (raw["delta_x2"][0], raw["delta_x2"][1]),
        "mass": raw["mass"],
        "sho_omega": raw["sho_omega"],
    }


def load_energy_eigenstates(path: Path) -> EnergyEigenstates:

    with path.open("r") as f:
        out: EnergyEigenstatesRaw = json.load(f)

        eigenvectors = np.array(out["eigenvectors_re"]) + 1j * np.array(
            out["eigenvectors_im"]
        )

        return {
            "eigenstate_config": eigenstates_config_from_raw(out["eigenstate_config"]),
            "eigenvalues": out["eigenvalues"],
            "eigenvectors": eigenvectors.tolist(),
            "kx_points": out["kx_points"],
            "ky_points": out["ky_points"],
        }


def load_energy_eigenstates_legacy(path: Path) -> EnergyEigenstates:
    class EigenstateConfigLegacy(TypedDict):
        resolution: List[int]
        sho_omega: float
        mass: float
        delta_x: float
        delta_y: float

    class EnergyEigenstatesRawLegacy(TypedDict):
        eigenstate_config: EigenstateConfigLegacy
        kx_points: List[float]
        ky_points: List[float]
        eigenvalues: List[float]
        eigenvectors_re: List[List[float]]
        eigenvectors_im: List[List[float]]

    def config_from_legacy(config: EigenstateConfigLegacy) -> EigenstateConfig:
        return {
            "resolution": (
                config["resolution"][0],
                config["resolution"][1],
                config["resolution"][2],
            ),
            "delta_x0": (config["delta_x"], 0),
            "delta_x1": (0, config["delta_y"]),
            "mass": config["mass"],
            "sho_omega": config["sho_omega"],
        }

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
        data: EnergyEigenstatesRaw = json.load(f)
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
    max_index: int = int(np.argmax(above_threshold) - 1) % z_points.shape[0]
    above_threshold = (z_indexes < min_z) & (z_points > fit_max_energy)

    # Search backwards, stops at the first above threshold
    min_index: int = (
        z_points.shape[0] - np.argmax(above_threshold[::-1])
    ) % z_points.shape[0]

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


def filter_eigenstates_band(
    eigenstates: EnergyEigenstates, n: int = 0
) -> EnergyEigenstates:
    kx_points = np.array(eigenstates["kx_points"])
    ky_points = np.array(eigenstates["ky_points"])
    eigenvalues = np.array(eigenstates["eigenvalues"])
    eigenvectors = np.array(eigenstates["eigenvectors"])
    kx_ky_points = np.unique(
        [(kx, ky) for (kx, ky) in zip(kx_points, ky_points)],
        axis=0,
    )
    grouped_eigenvalues = [
        eigenvalues[np.logical_and(kx == kx_points, ky == ky_points)]
        for (kx, ky) in kx_ky_points
    ]
    grouped_eigenvectors = [
        eigenvectors[np.logical_and(kx == kx_points, ky == ky_points)]
        for (kx, ky) in kx_ky_points
    ]
    band_index = [np.argpartition(v, n)[n] for v in grouped_eigenvalues]

    return {
        "eigenstate_config": eigenstates["eigenstate_config"],
        "eigenvalues": [
            float(vals[i]) for (i, vals) in zip(band_index, grouped_eigenvalues)
        ],
        "eigenvectors": [
            vals[i].tolist() for (i, vals) in zip(band_index, grouped_eigenvectors)
        ],
        "kx_points": kx_ky_points[:, 0].tolist(),
        "ky_points": kx_ky_points[:, 1].tolist(),
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


def get_brillouin_points_irreducible_config(
    config: EigenstateConfig, *, size: Tuple[int, int] = (4, 4), include_zero=True
):
    """
    If the eigenstate config is that of the irreducible unit cell
    we can use the dkx of the lattuice to generate the brillouin zone points
    """
    util = EigenstateConfigUtil(config)
    return get_points_in_brillouin_zone(
        util.dkx0, util.dkx1, size=size, include_zero=include_zero
    )
