import json
from pathlib import Path
from typing import List, Tuple, TypedDict

import numpy as np

from .sho_config import EigenstateConfig


class EnergyEigenstates(TypedDict):
    eigenstate_config: EigenstateConfig
    kx_points: List[float]
    ky_points: List[float]
    resolution: Tuple[int, int, int]
    eigenvalues: List[float]
    eigenvectors: List[List[complex]]


def save_energy_eigenstates(data: EnergyEigenstates, path: Path) -> None:
    with path.open("w") as f:
        json.dump(
            {
                "eigenstate_config": data["eigenstate_config"],
                "eigenvalues": data["eigenvalues"],
                "eigenvectors_re": np.real(data["eigenvectors"]).tolist(),
                "eigenvectors_im": np.imag(data["eigenvectors"]).tolist(),
                "kx_points": data["kx_points"],
                "ky_points": data["ky_points"],
                "resolution": data["resolution"],
            },
            f,
        )


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
    path: Path, kx: float, ky: float, eigenvalue: float, eigenvector: List[complex]
) -> None:
    with path.open("r") as f:
        data: EnergyEigenstates = json.load(f)
        data["kx_points"].append(kx)
        data["ky_points"].append(ky)
        data["eigenvalues"].append(eigenvalue)
        data["eigenvectors"].append(eigenvector)

    with path.open("w") as f:
        json.dump(data, f)


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
