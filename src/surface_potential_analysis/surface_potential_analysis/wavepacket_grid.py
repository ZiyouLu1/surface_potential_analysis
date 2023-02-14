import json
from pathlib import Path
from typing import List, Literal, Tuple, TypedDict

import numpy as np
import scipy
from numpy.typing import NDArray

from .energy_eigenstate import (
    EigenstateConfigUtil,
    EnergyEigenstates,
    get_eigenstate_list,
)
from .surface_config import (
    SurfaceConfig,
    get_surface_coordinates,
    get_surface_xy_points,
)


class WavepacketGrid(SurfaceConfig):
    z_points: List[float]
    points: List[List[List[complex]]]


class WavepacketGridRaw(TypedDict):
    delta_x0: List[float]
    delta_x1: List[float]
    z_points: List[float]
    real_points: List[List[List[complex]]]
    imag_points: List[List[List[complex]]]


def save_wavepacket_grid(data: WavepacketGrid, path: Path) -> None:
    with path.open("w") as f:
        out: WavepacketGridRaw = {
            "real_points": np.real(data["points"]).tolist(),
            "imag_points": np.imag(data["points"]).tolist(),
            "delta_x0": list(data["delta_x0"]),
            "delta_x1": list(data["delta_x1"]),
            "z_points": data["z_points"],
        }
        json.dump(out, f)


def load_wavepacket_grid(path: Path) -> WavepacketGrid:
    with path.open("r") as f:
        out: WavepacketGridRaw = json.load(f)
        points = np.array(out["real_points"]) + 1j * np.array(out["imag_points"])
        return {
            "points": points.tolist(),
            "delta_x0": (out["delta_x0"][0], out["delta_x0"][1]),
            "delta_x1": (out["delta_x1"][0], out["delta_x1"][1]),
            "z_points": out["z_points"],
        }


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
            "delta_x0": (out2["x_points"][-1] - out2["x_points"][0], 0),
            "delta_x1": (0, out2["y_points"][-1] - out2["y_points"][0]),
            "z_points": out2["z_points"],
        }


def reflect_wavepacket_in_axis(
    wavepacket: WavepacketGrid, axis: Literal[1, 2] = 1
) -> WavepacketGrid:
    points = np.array(wavepacket["points"])

    reflected_shape = (
        points.shape[0] * 2 - 1 if axis == 2 else points.shape[0],
        points.shape[1] * 2 - 1 if axis == 1 else points.shape[1],
        points.shape[2],
    )
    reflected_points = np.zeros(reflected_shape, dtype=complex)

    if axis == 2:
        reflected_points[points.shape[0] - 1 :, :] = points[:, :]
        reflected_points[: points.shape[0], :] = points[::-1, :]
    else:
        reflected_points[:, points.shape[1] - 1 :] = points[:, :]
        reflected_points[:, : points.shape[0]] = points[:, ::-1]

    return {
        "points": reflected_points.tolist(),
        "delta_x0": (
            (wavepacket["delta_x0"][0] * 2, wavepacket["delta_x0"][1] * 2)
            if axis == 2
            else wavepacket["delta_x0"]
        ),
        "delta_x1": (
            (wavepacket["delta_x1"][0] * 2, wavepacket["delta_x1"][1] * 2)
            if axis == 1
            else wavepacket["delta_x1"]
        ),
        "z_points": wavepacket["z_points"],
    }


def symmetrize_wavepacket_about_far_edge(wavepacket: WavepacketGrid) -> WavepacketGrid:

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
        "delta_x0": (wavepacket["delta_x0"][0] * 2, wavepacket["delta_x0"][1] * 2),
        "delta_x1": (wavepacket["delta_x1"][0] * 2, wavepacket["delta_x1"][1] * 2),
        "z_points": wavepacket["z_points"],
    }


def interpolate_wavepacket(
    data: WavepacketGrid, shape: Tuple[int, int, int] = (40, 40, 100)
) -> WavepacketGrid:

    if (data["delta_x0"][1] != 0) or (data["delta_x1"][0] != 0):
        raise AssertionError("Not orthogonal grid")

    points = np.array(data["points"])
    x_points = np.linspace(0, data["delta_x0"][0], points.shape[0], endpoint=False)
    y_points = np.linspace(0, data["delta_x1"][1], points.shape[1], endpoint=False)
    z_points = np.array(data["z_points"])

    interpolator = scipy.interpolate.RegularGridInterpolator(
        [x_points, y_points, z_points],
        np.real(points),
    )

    x_points = np.linspace(0, data["delta_x0"][0], shape[0], endpoint=False)
    y_points = np.linspace(0, data["delta_x1"][1], shape[1], endpoint=False)
    z_points = np.linspace(data["z_points"][0], data["z_points"][-1], shape[2])
    xt, yt, zt = np.meshgrid(x_points, y_points, z_points, indexing="ij")

    test_points = np.array([xt.ravel(), yt.ravel(), zt.ravel()]).T
    split = np.array_split(test_points, 1 + test_points.shape[0] // 100000)

    def interpolate_cubic(s):
        out = interpolator(s, method="cubic")
        return out

    points = np.concatenate([interpolate_cubic(s) for s in split])

    return {
        "points": points.reshape(*shape).tolist(),
        "delta_x0": data["delta_x0"],
        "delta_x1": data["delta_x1"],
        "z_points": data["z_points"],
    }


def calculate_volume_element(wavepacket: WavepacketGrid) -> float:
    xy_area = np.linalg.norm(np.cross(wavepacket["delta_x0"], wavepacket["delta_x1"]))
    volume = xy_area * (wavepacket["z_points"][-1] - wavepacket["z_points"][0])
    n_points = np.product(np.array(wavepacket["points"]).shape)
    return float(volume / n_points)


def mask_negative_wavepacket(wavepacket: WavepacketGrid) -> WavepacketGrid:
    points = np.real_if_close(wavepacket["points"])
    points[points < 0] = 0
    return {
        "points": points.tolist(),
        "delta_x0": wavepacket["delta_x0"],
        "delta_x1": wavepacket["delta_x1"],
        "z_points": wavepacket["z_points"],
    }


def get_wavepacket_grid_xy_points(grid: WavepacketGrid) -> NDArray:
    points = np.real(grid["points"])
    return get_surface_xy_points(grid, (points.shape[0], points.shape[1]))


def get_wavepacket_grid_coordinates(
    grid: WavepacketGrid, *, offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> NDArray:
    points = np.real(grid["points"])
    z_points = np.linspace(
        offset[2], grid["delta_z"] + offset[2], points.shape[2]
    ).tolist()
    shape = (points.shape[0], points.shape[1])
    xy_offset = (offset[0], offset[1])
    return get_surface_coordinates(grid, shape, z_points, offset=xy_offset)


def calculate_wavepacket_grid_copper(
    eigenstates: EnergyEigenstates,
    *,
    shape: Tuple[int, int, int] = (49, 49, 21),
):

    util = EigenstateConfigUtil(eigenstates["eigenstate_config"])
    return calculate_wavepacket_grid(
        eigenstates,
        delta_x0=(2 * util.delta_x0[0], 2 * util.delta_x0[1]),
        delta_x1=(2 * util.delta_x1[0], 2 * util.delta_x1[1]),
        z_points=np.linspace(
            -2 * util.characteristic_z, 2 * util.characteristic_z, shape[2]
        ).tolist(),
        shape=shape[0:2],
        offset=(-util.delta_x0[0], -util.delta_x1[1], -2 * util.characteristic_z),
    )


def calculate_wavepacket_grid_new(
    eigenstates: EnergyEigenstates, z_points: List[float]
):
    """
    Since we generate the energy eigenstates from a limited number of k states in
    the x0, x1 direction we do not have the information necessary to properly interpolate

    """
    util = EigenstateConfigUtil(eigenstates["eigenstate_config"])
    shape = (2 * util.resolution[0], 2 * util.resolution[1], len(z_points))
    print(shape)
    grid: WavepacketGrid = {
        "delta_x0": (
            2 * eigenstates["eigenstate_config"]["delta_x0"][0],
            2 * eigenstates["eigenstate_config"]["delta_x0"][1],
        ),
        "delta_x1": (
            2 * eigenstates["eigenstate_config"]["delta_x1"][0],
            2 * eigenstates["eigenstate_config"]["delta_x1"][1],
        ),
        "z_points": z_points,
        "points": np.empty(shape).tolist(),
    }
    coordinates = get_surface_coordinates(grid, (shape[0], shape[1]), z_points)
    coordinates_flat = coordinates.reshape(-1, 3)

    points = np.zeros(shape, dtype=complex)
    for eigenstate in get_eigenstate_list(eigenstates):
        print(eigenstate["kx"], eigenstate["ky"])
        wfn = util.calculate_wavefunction_fast(eigenstate, coordinates_flat)
        points += wfn.reshape(shape) / len(eigenstates["eigenvectors"])

    grid["points"] = points.tolist()
    return grid


def calculate_wavepacket_grid(
    eigenstates: EnergyEigenstates,
    delta_x0: Tuple[float, float],
    delta_x1: Tuple[float, float],
    z_points: List[float],
    shape: Tuple[int, int] = (49, 49),
    *,
    offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> WavepacketGrid:

    util = EigenstateConfigUtil(eigenstates["eigenstate_config"])
    grid: WavepacketGrid = {
        "delta_x0": delta_x0,
        "delta_x1": delta_x1,
        "z_points": z_points,
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
        wfn = util.calculate_wavefunction_fast(eigenstate, coordinates_flat)
        points += wfn.reshape(shape) / len(eigenstates["eigenvectors"])

    grid["points"] = points.tolist()
    return grid
