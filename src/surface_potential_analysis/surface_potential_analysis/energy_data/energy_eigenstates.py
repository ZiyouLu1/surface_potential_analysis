import json
from pathlib import Path
from typing import List, Tuple, TypedDict

from .sho_config import EigenstateConfig


class EnergyEigenstates(TypedDict):
    eigenstate_config: EigenstateConfig
    kx_points: List[float]
    ky_points: List[float]
    resolution: Tuple[int, int, int]
    eigenvalues: List[float]
    eigenvectors: List[List[float]]


def save_energy_eigenstates(data: EnergyEigenstates, path: Path) -> None:
    with path.open("w") as f:
        json.dump(data, f)


def load_energy_eigenstates(path: Path) -> EnergyEigenstates:
    with path.open("r") as f:
        return json.load(f)


def append_energy_eigenstates(
    path: Path, kx: float, ky: float, eigenvalue: float, eigenvector: List[float]
) -> None:
    with path.open("r") as f:
        data: EnergyEigenstates = json.load(f)
        data["kx_points"].append(kx)
        data["ky_points"].append(ky)
        data["eigenvalues"].append(eigenvalue)
        data["eigenvectors"].append(eigenvector)

    with path.open("w") as f:
        json.dump(data, f)


# TODO: remove
# def mirror_energy_eigenstates(data: EnergyEigenstates) -> EnergyEigenstates:
#     kx_points = np.concatenate(
#         [np.tile(data["kx_points"], 2), -np.tile(data["kx_points"], 2)]
#     )
#     ky_points = np.tile(
#         np.concatenate([data["ky_points"], -np.array(data["ky_points"])]), 2
#     )

#     return {
#         "eigenvalues": np.tile(data["eigenvalues"], 4).tolist(),
#         "eigenvectors": np.tile(data["eigenvectors"], 4).tolist(),
#         "resolution": data["resolution"],
#         "kx_points": kx_points.tolist(),
#         "ky_points": ky_points.tolist(),
#     }
