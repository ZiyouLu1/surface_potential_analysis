import json
from pathlib import Path
from typing import List, Tuple, TypedDict

import numpy as np
import scipy
from numpy.typing import NDArray

from surface_potential_analysis.energy_data import (
    get_energy_grid_coordinates,
    get_energy_grid_xy_points,
)

from .energy_eigenstate import (
    EigenstateConfigUtil,
    EnergyEigenstates,
    get_eigenstate_list,
)


class WavepacketGrid(TypedDict):
    delta_x1: Tuple[float, float]
    delta_x2: Tuple[float, float]
    delta_z: float
    points: List[List[List[complex]]]


def save_wavepacket_grid(data: WavepacketGrid, path: Path) -> None:
    with path.open("w") as f:
        json.dump(
            {
                "real_points": np.real(data["points"]).tolist(),
                "imag_points": np.imag(data["points"]).tolist(),
                "delta_x1": data["delta_x1"],
                "delta_x2": data["delta_x2"],
                "delta_z": data["delta_z"],
            },
            f,
        )


def load_wavepacket_grid_legacy(path: Path) -> WavepacketGrid:
    class WavepacketGridLegacy(TypedDict):
        x_points: List[float]
        y_points: List[float]
        z_points: List[float]
        points: List[List[List[complex]]]

    with path.open("r") as f:
        out = json.load(f)
        points = np.array(out["real_points"]) + 1j * np.array(out["imag_points"])
        out["points"] = points.tolist()

        out2: WavepacketGridLegacy = out

        return {
            "points": out2["points"],
            "delta_x1": (out2["x_points"][-1] - out2["x_points"][0], 0),
            "delta_x2": (0, out2["y_points"][-1] - out2["y_points"][0]),
            "delta_z": out2["z_points"][-1] - out2["z_points"][0],
        }


def symmetrize_wavepacket(wavepacket: WavepacketGrid) -> WavepacketGrid:

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
        "delta_x1": (wavepacket["delta_x1"][0] * 2, wavepacket["delta_x1"][1] * 2),
        "delta_x2": (wavepacket["delta_x2"][0] * 2, wavepacket["delta_x2"][1] * 2),
        "delta_z": wavepacket["delta_z"],
    }


def interpolate_wavepacket(
    data: WavepacketGrid, shape: Tuple[int, int, int] = (40, 40, 100)
) -> WavepacketGrid:

    if (data["delta_x1"][1] != 0) or (data["delta_x2"][0] != 0):
        raise AssertionError("Not orthogonal grid")

    points = np.array(data["points"])
    x_points = np.linspace(0, data["delta_x1"][0], points.shape[0], endpoint=False)
    y_points = np.linspace(0, data["delta_x2"][1], points.shape[1], endpoint=False)
    z_points = np.linspace(0, data["delta_z"], points.shape[1])

    interpolator = scipy.interpolate.RegularGridInterpolator(
        [x_points, y_points, z_points],
        np.real(points),
    )

    x_points = np.linspace(0, data["delta_x1"][0], shape[0], endpoint=False)
    y_points = np.linspace(0, data["delta_x2"][1], shape[1], endpoint=False)
    z_points = np.linspace(0, data["delta_z"], shape[2])
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
        "delta_x1": data["delta_x1"],
        "delta_x2": data["delta_x2"],
        "delta_z": data["delta_z"],
    }


def calculate_volume_element(wavepacket: WavepacketGrid) -> float:
    xy_area = np.linalg.norm(np.cross(wavepacket["delta_x1"], wavepacket["delta_x2"]))
    volume = xy_area * wavepacket["delta_z"]
    n_points = np.product(np.array(wavepacket["points"]).shape)
    return float(volume / n_points)


def mask_negative_wavepacket(wavepacket: WavepacketGrid) -> WavepacketGrid:
    points = np.real_if_close(wavepacket["points"])
    points[points < 0] = 0
    return {
        "points": points.tolist(),
        "delta_x1": wavepacket["delta_x1"],
        "delta_x2": wavepacket["delta_x2"],
        "delta_z": wavepacket["delta_z"],
    }


def get_wavepacket_grid_xy_points(grid: WavepacketGrid) -> NDArray:
    points = np.real(grid["points"])
    return get_energy_grid_xy_points(
        {
            "delta_x1": grid["delta_x1"],
            "delta_x2": grid["delta_x2"],
            "points": points.tolist(),
            "z_points": np.linspace(0, grid["delta_z"], points.shape[2]).tolist(),
        }
    )


def get_wavepacket_grid_coordinates(
    grid: WavepacketGrid, *, offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> NDArray:
    points = np.real(grid["points"])
    print(points.shape)
    return get_energy_grid_coordinates(
        {
            "delta_x1": grid["delta_x1"],
            "delta_x2": grid["delta_x2"],
            "points": points.tolist(),
            "z_points": np.linspace(0, grid["delta_z"], points.shape[2]).tolist(),
        },
        offset=offset,
    )


def calculate_wavepacket_grid_copper(
    eigenstates: EnergyEigenstates,
    *,
    cutoff: int | None = None,
    shape: Tuple[int, int, int] = (49, 49, 21),
):

    util = EigenstateConfigUtil(eigenstates["eigenstate_config"])
    return calculate_wavepacket_grid(
        eigenstates,
        delta_x1=(2 * util.delta_x1[0], 2 * util.delta_x1[1]),
        delta_x2=(2 * util.delta_x2[0], 2 * util.delta_x2[1]),
        delta_z=4 * util.characteristic_z,
        shape=shape,
        cutoff=cutoff,
        offset=(-util.delta_x1[0], -util.delta_x2[1], -2 * util.characteristic_z),
    )


def calculate_wavepacket_grid(
    eigenstates: EnergyEigenstates,
    delta_x1: Tuple[float, float],
    delta_x2: Tuple[float, float],
    delta_z: float,
    shape: Tuple[int, int, int] = (49, 49, 21),
    *,
    offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    cutoff: int | None = None,
) -> WavepacketGrid:

    util = EigenstateConfigUtil(eigenstates["eigenstate_config"])
    grid: WavepacketGrid = {
        "delta_x1": delta_x1,
        "delta_x2": delta_x2,
        "delta_z": delta_z,
        "points": np.zeros(shape).tolist(),
    }
    coordinates = get_wavepacket_grid_coordinates(grid, offset=offset)
    coordinates_flat = coordinates.reshape(-1, 3)

    if not np.array_equal(
        coordinates[:, :, :, 0], coordinates_flat[:, 0].reshape(shape)
    ):
        raise AssertionError("Error unraveling points")

    points = np.zeros(shape, dtype=complex)
    for eigenstate in get_eigenstate_list(eigenstates):
        print(eigenstate["kx"], eigenstate["ky"])
        wfn = (
            util.calculate_wavefunction_slow(
                eigenstate,
                coordinates_flat,
                cutoff=cutoff,
            )
            if cutoff is not None
            else util.calculate_wavefunction_fast(
                eigenstate,
                coordinates_flat,
            )
        )
        points += wfn.reshape(shape) / len(eigenstates["eigenvectors"])

    grid["points"] = points.tolist()
    return grid
