import json
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import scipy
import scipy.fft
from numpy.typing import NDArray

from surface_potential_analysis.interpolation import interpolate_points_fftn

from .energy_eigenstate import (
    EigenstateConfigUtil,
    EnergyEigenstates,
    get_eigenstate_list,
)
from .surface_config import SurfaceConfig, get_surface_coordinates


class WavepacketGrid(SurfaceConfig):
    z_points: list[float]
    points: list[list[list[complex]]]


class WavepacketGridRaw(TypedDict):
    delta_x0: list[float]
    delta_x1: list[float]
    z_points: list[float]
    real_points: list[list[list[complex]]]
    imag_points: list[list[list[complex]]]


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
        x_points: list[float]
        y_points: list[float]
        z_points: list[float]
        points: list[list[list[complex]]]

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


def interpolate_wavepacket(
    data: WavepacketGrid, shape: tuple[int, int, int] = (40, 40, 100)
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
    n_points = np.product(np.shape(wavepacket["points"]))
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


def get_wavepacket_grid_coordinates(
    grid: WavepacketGrid, *, offset: tuple[float, float] = (0.0, 0.0)
) -> NDArray:
    points = np.real(grid["points"])
    z_points = np.array(grid["z_points"]).tolist()
    shape = (points.shape[0], points.shape[1])
    return get_surface_coordinates(grid, shape, z_points, offset=offset)


def calculate_wavepacket_grid_copper(
    eigenstates: EnergyEigenstates,
    *,
    shape: tuple[int, int, int] = (49, 49, 21),
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
        offset=(-util.delta_x0[0], -util.delta_x1[1]),
    )


def calculate_wavepacket_grid_fourier(
    eigenstates: EnergyEigenstates,
    z_points: list[float],
    x0_lim: tuple[int, int] = (0, 1),
    x1_lim: tuple[int, int] = (0, 1),
):
    """
    Since we generate the energy eigenstates from a limited number of k states in
    the x0, x1 direction we do not have the information necessary to properly interpolate
    to spacing on a finer mesh than the fourier transform
    """
    util = EigenstateConfigUtil(eigenstates["eigenstate_config"])
    nx0 = x0_lim[1] - x0_lim[0]
    nx1 = x1_lim[1] - x1_lim[0]
    shape = (util.resolution[0] * nx0, util.resolution[1] * nx1, len(z_points))

    grid: WavepacketGrid = {
        "delta_x0": (
            eigenstates["eigenstate_config"]["delta_x0"][0] * nx0,
            eigenstates["eigenstate_config"]["delta_x0"][1] * nx0,
        ),
        "delta_x1": (
            eigenstates["eigenstate_config"]["delta_x1"][0] * nx1,
            eigenstates["eigenstate_config"]["delta_x1"][1] * nx1,
        ),
        "z_points": z_points,
        "points": np.empty(shape).tolist(),
    }

    points = np.zeros(shape, dtype=complex)
    # Ns = int(np.sqrt(len(eigenstates["eigenvectors"])))
    for (i, eigenstate) in enumerate(get_eigenstate_list(eigenstates)):
        print(eigenstate["kx"], eigenstate["ky"])
        wfn = util.calculate_wavefunction_slow_grid_fourier(
            eigenstate, z_points, x0_lim, x1_lim
        )
        # wfn = util.calculate_wavefunction_slow_grid_fourier_exact_phase(
        #     eigenstate["eigenvector"],
        #     (i // Ns, i % Ns),
        #     (Ns, Ns),
        #     z_points,
        #     x0_lim,
        #     x1_lim,
        # )
        points += wfn / np.sqrt(len(eigenstates["eigenvectors"]))

    grid["points"] = points.tolist()
    return grid


def calculate_wavepacket_grid_fourier_fourier(
    eigenstates: EnergyEigenstates,
    z_points: list[float],
    x0_lim: tuple[int, int] = (0, 1),
    x1_lim: tuple[int, int] = (0, 1),
) -> WavepacketGrid:
    """
    Calculate the wavepacket in the first unit cell without explicitly referencing the
    individual frequencies

    Since we generate the energy eigenstates from a limited number of k states in
    the x0, x1 direction we do not have the information necessary to properly interpolate
    to spacing on a finer mesh than the fourier transform


    Parameters
    ----------
    eigenstates : EnergyEigenstates
        _description_
    z_points : list[float]
        _description_

    Returns
    -------
    WavepacketGrid
        _description_
    """
    util = EigenstateConfigUtil(eigenstates["eigenstate_config"])
    nx0 = x0_lim[1] - x0_lim[0]
    nx1 = x1_lim[1] - x1_lim[0]
    shape = (util.resolution[0] * nx0, util.resolution[1] * nx1, len(z_points))

    # Assume the grid is square
    n_sample = int(np.sqrt(np.shape(eigenstates["eigenvectors"])[0]))
    eigenvectors = np.reshape(eigenstates["eigenvectors"], (n_sample, n_sample, -1))

    def calculate_bloch_wfn(eigenvector):
        return util.calculate_bloch_wavefunction_fourier(eigenvector, z_points)

    bloch_wfns = np.apply_along_axis(calculate_bloch_wfn, -1, eigenvectors)

    # Calculate the inverse fourier transform of the wavefunctions
    # A[m] = 1/Ns Sum k=0..(Nx0-1) U_k exp(2j pi k m / Ns)
    # for m = 0,1..(Ns-1), where U_k is the value of the bloch wavefunction
    # for the kth component in the wavepacket grid
    wavepacket = scipy.fft.ifft2(bloch_wfns, axes=(0, 1))
    # Interpolate the fourier transform to get the relevant points on the
    # overall wavefunction. We want to calculate the phased sum of the bloch
    # wavefunctions
    # phi[xi] = 1/Ns Sum k=0..(Nx0-1) U_k exp(2j pi k (x0i) / delta_x0 Ns)
    # where 2pi k / delta_x0 is the frequency of the kth bloch wavefunction
    # Since we need the points at the fractional frequencies
    # m = x0i/delta_x0 = 0..(Nx0-1)/Nx0
    # we need the Nx0 points between the 0th and 1st element in the fourier
    # transform, which before the interpolation just contains the frequencies m = 0,1..(Ns-1)
    interpolated = interpolate_points_fftn(
        wavepacket,
        (
            wavepacket.shape[0] * wavepacket.shape[2],
            wavepacket.shape[1] * wavepacket.shape[3],
        ),
        axes=(0, 1),
    )
    # Sum over the diagonals, taking the relevant frequency component
    # for each point xi
    summed = np.zeros(shape, dtype=complex)
    for ix0 in range(x0_lim[0] * util.Nkx0, x0_lim[1] * util.Nkx0):
        for ix1 in range(x1_lim[0] * util.Nkx1, x1_lim[1] * util.Nkx1):
            summed[ix0, ix1] = interpolated[ix0, ix1, ix0 % util.Nkx0, ix1 % util.Nkx1]
    # since the negative ix0 wrap around we need to shift the large x components back
    points = np.fft.fftshift(summed, axes=(0, 1)).tolist()

    return {
        "delta_x0": (
            eigenstates["eigenstate_config"]["delta_x0"][0] * nx0,
            eigenstates["eigenstate_config"]["delta_x0"][1] * nx0,
        ),
        "delta_x1": (
            eigenstates["eigenstate_config"]["delta_x1"][0] * nx1,
            eigenstates["eigenstate_config"]["delta_x1"][1] * nx1,
        ),
        "z_points": z_points,
        "points": points,
    }


def calculate_wavepacket_grid(
    eigenstates: EnergyEigenstates,
    delta_x0: tuple[float, float],
    delta_x1: tuple[float, float],
    z_points: list[float],
    shape: tuple[int, int] = (49, 49),
    *,
    offset: tuple[float, float] = (0.0, 0.0),
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
        points += wfn.reshape(shape) / np.sqrt(len(eigenstates["eigenvectors"]))
        # wfn = util.calculate_wavefunction_slow_grid_fourier(eigenstate, z_points)
        # points += wfn / len(eigenstates["eigenvectors"])

    grid["points"] = points.tolist()
    return grid


def calculate_inner_product(grid0: WavepacketGrid, grid1: WavepacketGrid) -> complex:
    """
    Calculates the inner product of two wavepacket grids
    equal to the (0,0) point on the overlap transform

    Parameters
    ----------
    grid0 : WavepacketGrid
    grid1 : WavepacketGrid

    Returns
    -------
    complex
        The inner product of the two wavepacket grids
    """
    delta_z = grid0["z_points"][-1] - grid0["z_points"][0]
    N = np.prod(np.shape(grid0["points"]))

    return np.sum(np.multiply(np.conj(grid0["points"]), grid1["points"])) * delta_z / N


def calculate_normalisation(grid: WavepacketGrid) -> float:
    # Should always be real!
    return np.real(calculate_inner_product(grid, grid))
