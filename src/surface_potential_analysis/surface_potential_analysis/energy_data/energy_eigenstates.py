import json
from pathlib import Path
from typing import List, Tuple, TypedDict

import numpy as np
import scipy

from .sho_config import EigenstateConfig


class Eigenstate(TypedDict):
    kx: float
    ky: float
    eigenvector: List[complex]


class EnergyEigenstates(TypedDict):
    eigenstate_config: EigenstateConfig
    kx_points: List[float]
    ky_points: List[float]
    resolution: Tuple[int, int, int]
    eigenvalues: List[float]
    eigenvectors: List[List[complex]]


class EnergyEigenstatesRaw(TypedDict):
    eigenstate_config: EigenstateConfig
    kx_points: List[float]
    ky_points: List[float]
    resolution: Tuple[int, int, int]
    eigenvalues: List[float]
    eigenvectors_re: List[List[float]]
    eigenvectors_im: List[List[float]]


def save_energy_eigenstates(data: EnergyEigenstates, path: Path) -> None:
    with path.open("w") as f:
        out: EnergyEigenstatesRaw = {
            "eigenstate_config": data["eigenstate_config"],
            "eigenvalues": data["eigenvalues"],
            "eigenvectors_re": np.real(data["eigenvectors"]).tolist(),
            "eigenvectors_im": np.imag(data["eigenvectors"]).tolist(),
            "kx_points": data["kx_points"],
            "ky_points": data["ky_points"],
            "resolution": data["resolution"],
        }
        json.dump(out, f)


def load_energy_eigenstates(path: Path) -> EnergyEigenstates:
    with path.open("r") as f:
        out = json.load(f)

        eigenvectors = (
            np.array(out["eigenvectors"])
            if out.get("eigenvectors_im", None) is None
            else np.array(out["eigenvectors_re"])
            + 1j * np.array(out["eigenvectors_im"])
        )

        return {
            "eigenstate_config": out["eigenstate_config"],
            "eigenvalues": out["eigenvalues"],
            "eigenvectors": eigenvectors.tolist(),
            "kx_points": out["kx_points"],
            "ky_points": out["ky_points"],
            "resolution": out["resolution"],
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


class WavepacketGrid(TypedDict):
    x_points: List[float]
    y_points: List[float]
    z_points: List[float]
    points: List[List[List[complex]]]


def save_wavepacket_grid(data: WavepacketGrid, path: Path) -> None:
    with path.open("w") as f:
        json.dump(
            {
                "real_points": np.real(data["points"]).tolist(),
                "imag_points": np.imag(data["points"]).tolist(),
                "x_points": data["x_points"],
                "y_points": data["y_points"],
                "z_points": data["z_points"],
            },
            f,
        )


def load_wavepacket_grid(path: Path) -> WavepacketGrid:
    with path.open("r") as f:
        out = json.load(f)
        points = np.array(out["real_points"]) + 1j * np.array(out["imag_points"])

        return {
            "points": points.tolist(),
            "x_points": out["x_points"],
            "y_points": out["y_points"],
            "z_points": out["z_points"],
        }


def symmetrize_wavepacket(wavepacket: WavepacketGrid) -> WavepacketGrid:
    x_points = np.array(wavepacket["x_points"])
    y_points = np.array(wavepacket["y_points"])

    reflected_x_points = ((2 * x_points[-1]) - x_points)[:-1]
    reflected_y_points = ((2 * y_points[-1]) - y_points)[:-1]
    points = np.array(wavepacket["points"])

    reflected_shape = (
        points.shape[0] * 2 - 1,
        points.shape[1] * 2 - 1,
        points.shape[2],
    )
    reflected_points = np.zeros(reflected_shape, dtype=complex)
    reflected_points[: points.shape[0], : points.shape[1]] = points[:, :]
    reflected_points[points.shape[0] - 1 :, : points.shape[1]] = points[::-1, :]
    reflected_points[: points.shape[0], points.shape[1] - 1 :] = points[:, ::-1]
    reflected_points[points.shape[0] - 1 :, points.shape[1] - 1 :] = points[::-1, ::-1]

    return {
        "points": reflected_points.tolist(),
        "x_points": np.sort(np.concatenate([x_points, reflected_x_points])).tolist(),
        "y_points": np.sort(np.concatenate([y_points, reflected_y_points])).tolist(),
        "z_points": wavepacket["z_points"],
    }


def sort_wavepacket(wavepacket: WavepacketGrid) -> WavepacketGrid:
    x_sort = np.argsort(wavepacket["x_points"])
    y_sort = np.argsort(wavepacket["y_points"])
    z_sort = np.argsort(wavepacket["z_points"])

    x_points = np.array(wavepacket["x_points"])[x_sort]
    y_points = np.array(wavepacket["y_points"])[y_sort]
    z_points = np.array(wavepacket["z_points"])[z_sort]

    points = np.array(wavepacket["points"])[x_sort, :, :][:, y_sort, :][:, :, z_sort]

    return {
        "points": points.tolist(),
        "x_points": x_points.tolist(),
        "y_points": y_points.tolist(),
        "z_points": z_points.tolist(),
    }


def interpolate_wavepacket(
    data: WavepacketGrid, shape: Tuple[int, int, int] = (40, 40, 100)
) -> WavepacketGrid:
    sorted = sort_wavepacket(data)

    interpolator = scipy.interpolate.RegularGridInterpolator(
        [
            sorted["x_points"],
            sorted["y_points"],
            sorted["z_points"],
        ],
        np.real(sorted["points"]),
    )

    x_points = np.linspace(
        sorted["x_points"][0], sorted["x_points"][-1], shape[0], endpoint=False
    )
    y_points = np.linspace(
        sorted["y_points"][0], sorted["y_points"][-1], shape[1], endpoint=False
    )
    z_points = list(
        np.linspace(sorted["x_points"][0], sorted["z_points"][-1], shape[2])
    )
    xt, yt, zt = np.meshgrid(x_points, y_points, z_points, indexing="ij")
    test_points = np.array([xt.ravel(), yt.ravel(), zt.ravel()]).T
    points = np.zeros_like(test_points)

    print(test_points.shape[0], test_points.shape[0] // 100000)
    split = np.array_split(test_points, 1 + test_points.shape[0] // 100000)
    print(len(split))

    def interpolate_cubic(s):
        out = interpolator(s, method="cubic")
        print("done")
        return out

    points = np.concatenate([interpolate_cubic(s) for s in split])

    return {
        "points": points.reshape(*shape).tolist(),
        "x_points": x_points.tolist(),
        "y_points": y_points.tolist(),
        "z_points": z_points,
    }


def calculate_volume_element(wavepacket: WavepacketGrid):
    dx = wavepacket["x_points"][1] - wavepacket["x_points"][0]
    dy = wavepacket["y_points"][1] - wavepacket["y_points"][0]
    dz = wavepacket["z_points"][1] - wavepacket["z_points"][0]
    return dx * dy * dz


def mask_negative_wavepacket(wavepacket: WavepacketGrid) -> WavepacketGrid:
    points = np.real_if_close(wavepacket["points"])
    points[points < 0] = 0
    return {
        "points": points.tolist(),
        "x_points": wavepacket["x_points"],
        "y_points": wavepacket["y_points"],
        "z_points": wavepacket["z_points"],
    }
