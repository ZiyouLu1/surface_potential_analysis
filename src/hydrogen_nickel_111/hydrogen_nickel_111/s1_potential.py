from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np
import scipy.interpolate
from surface_potential_analysis.axis.axis import (
    FundamentalPositionBasis,
    FundamentalPositionBasis3d,
)
from surface_potential_analysis.axis.stacked_axis import (
    StackedBasis,
    StackedBasisLike,
)
from surface_potential_analysis.axis.util import (
    BasisUtil,
)
from surface_potential_analysis.potential import (
    PointPotential3d,
    UnevenPotential3d,
    interpolate_uneven_potential,
    load_point_potential_json,
    normalize_potential,
    truncate_potential,
    undo_truncate_potential,
)
from surface_potential_analysis.potential.potential import (
    Potential,
    UnevenPotential3dZBasis,
)
from surface_potential_analysis.stacked_basis.build import (
    position_basis_3d_from_shape,
)
from surface_potential_analysis.util.decorators import npy_cached
from surface_potential_analysis.util.interpolation import (
    interpolate_points_along_axis_spline,
)

from .surface_data import get_data_path

if TYPE_CHECKING:
    from pathlib import Path

    from surface_potential_analysis.axis.axis_like import AxisVector2d


def load_raw_potential_points() -> PointPotential3d[Any]:
    path = get_data_path("raw_data.json")
    return load_point_potential_json(path)


@npy_cached(
    get_data_path("potential/raw_potential_reciprocal_grid.npy"), load_pickle=True
)
def get_raw_potential_reciprocal_grid() -> UnevenPotential3d[Any, Any, Any]:
    data = load_raw_potential_points()
    x_points = data["x_points"]
    y_points = data["y_points"]
    z_points = data["z_points"]

    x_c = np.sort(np.unique(x_points))
    y_c = np.sort(np.unique(y_points))
    z_c = np.sort(np.unique(z_points))
    points = np.array(data["points"])

    is_top = np.logical_and(x_points == x_c[0], y_points == y_c[0])
    top_points = [points[np.logical_and(is_top, z_points == z)][0] for z in z_c]

    is_top_hcp = np.logical_and(x_points == x_c[0], y_points == y_c[2])
    top_hcp_points = [points[np.logical_and(is_top_hcp, z_points == z)][0] for z in z_c]

    is_hcp = np.logical_and(x_points == x_c[0], y_points == y_c[4])
    hcp_points = [points[np.logical_and(is_hcp, z_points == z)][0] for z in z_c]

    is_top_fcc = np.logical_and(x_points == x_c[1], y_points == y_c[1])
    top_fcc_points = [points[np.logical_and(is_top_fcc, z_points == z)][0] for z in z_c]

    is_fcc_hcp = np.logical_and(x_points == x_c[1], y_points == y_c[3])
    fcc_hcp_points = [points[np.logical_and(is_fcc_hcp, z_points == z)][0] for z in z_c]

    is_fcc = np.logical_and(x_points == x_c[2], y_points == y_c[2])
    fcc_points = [points[np.logical_and(is_fcc, z_points == z)][0] for z in z_c]

    # Turns out we don't have enough points to produce an 'out' grid in real space.
    # We therefore have to use the reciprocal grid of points
    hh = hcp_points
    hf = fcc_hcp_points
    ff = fcc_points
    tf = top_fcc_points
    tt = top_points
    th = top_hcp_points

    reciprocal_points = np.array(
        [
            [ff, tf, tt, th, hh, hf],
            [hf, th, tf, hf, th, tf],
            [hh, hf, ff, tf, tt, th],
            [th, tf, hf, th, tf, hf],
            [tt, th, hh, hf, ff, tf],
            [tf, hf, th, tf, hf, th],
        ]
    )

    length = np.max(y_points) - np.min(y_points)  # type: ignore[operator]
    return {
        "basis": StackedBasis(
            FundamentalPositionBasis(
                np.array([3 * length * (np.sqrt(3) / 2), 3 * length * (1 / 2)]),
                reciprocal_points.shape[0],
            ),
            FundamentalPositionBasis(
                np.array([0, 3 * length]), reciprocal_points.shape[1]
            ),
            UnevenPotential3dZBasis(z_c),
        ),
        "data": reciprocal_points.reshape(-1),
    }


def get_coordinate_fractions(
    vec0: tuple[float, float],
    vec1: tuple[float, float],
    coordinates: np.ndarray[Any, Any],
) -> np.ndarray[tuple[int, ...], np.dtype[np.float_]]:
    out: list[list[float]] = []
    for coord in coordinates:
        a = np.array(
            [
                [vec0[0], vec1[0]],
                [vec0[1], vec1[1]],
            ],
            dtype=np.float_,
        )
        fraction = np.linalg.solve(a, [coord[0], coord[1]])
        out.append([fraction[0], fraction[1]])
    return np.array(out)  # type: ignore[no-any-return]


_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)


def interpolate_points_fourier_nickel(  # noqa: PLR0913
    points: np.ndarray[tuple[int, int], np.dtype[np.complex_]],
    delta_x0_reciprocal: AxisVector2d,
    delta_x1_reciprocal: AxisVector2d,
    delta_x0_real: tuple[float, float],
    delta_x1_real: tuple[float, float],
    shape: tuple[_L0Inv, _L1Inv],
) -> np.ndarray[tuple[_L0Inv, _L1Inv], np.dtype[np.float_]]:
    """
    Interpolate a grid of points with the given shape into the real spacing.

    Given a uniform grid of points in the reciprocal spacing interpolate
    a grid of points with the given shape into the real spacing using the fourier transform.
    """
    ft_potential = np.fft.fft2(points, axes=(0, 1), norm="forward")
    old_basis = position_basis_3d_from_shape(
        shape=(points.shape[0], points.shape[1], 1)  # type: ignore[arg-type]
    )
    old_basis_util = BasisUtil(old_basis)
    nk_points_stacked = np.array(old_basis_util.fundamental_stacked_nk_points).reshape(
        3, *old_basis_util.shape
    )[0:2, :, :, 0]
    ft_phases = 2 * np.pi * np.moveaxis(nk_points_stacked, 0, -1)
    # List of [x1_frac, x2_frac] for the interpolated grid
    basis = StackedBasis(
        FundamentalPositionBasis(
            np.array([delta_x0_real[0], delta_x0_real[1], 1]), shape[0]
        ),
        FundamentalPositionBasis(
            np.array([delta_x1_real[0], delta_x1_real[1], 1]),
            shape[1],
        ),
        FundamentalPositionBasis(np.array([0, 0, 1]), 1),
    )
    util = BasisUtil(basis)
    coordinates = util.fundamental_x_points_stacked[0:2].T

    fractions = get_coordinate_fractions(
        (delta_x0_reciprocal.item(0), delta_x0_reciprocal.item(1)),
        (delta_x1_reciprocal.item(0), delta_x1_reciprocal.item(1)),
        coordinates,
    )

    # List of (List of list of [x1_phase, x2_phase] for the interpolated grid)
    interpolated_phases = np.multiply(
        fractions[:, np.newaxis, np.newaxis, :],
        ft_phases[np.newaxis, :, :, :],
    )
    # Sum over phase from x and y, raise to exp(-i * phi)
    summed_phases = np.exp(1j * np.sum(interpolated_phases, axis=-1))
    # Multiply the exponential by the prefactor form the fourier transform
    # Add the contribution from each ik1, ik2
    interpolated_points = np.sum(
        np.multiply(ft_potential[np.newaxis, :, :], summed_phases), axis=(1, 2)
    )
    np.testing.assert_array_almost_equal(
        interpolated_points, np.abs(interpolated_points)
    )
    return np.abs(interpolated_points).reshape(shape)  # type: ignore[no-any-return]


def interpolate_energy_grid_xy_fourier_nickel(
    data: UnevenPotential3d[Any, Any, Any],
    delta_x0_real: tuple[float, float],
    delta_x1_real: tuple[float, float],
    shape: tuple[_L0Inv, _L1Inv] = (40, 40),  # type: ignore[assignment]
) -> UnevenPotential3d[_L0Inv, _L1Inv, Any]:
    """Make use of a fourier transform to increase the number of points in the xy plane of the energy grid."""
    old_points = np.array(data["data"]).reshape(data["basis"].shape)
    points = np.empty((shape[0], shape[1], old_points.shape[2]))

    for iz in range(old_points.shape[2]):
        points[:, :, iz] = interpolate_points_fourier_nickel(
            old_points[:, :, iz],
            data["basis"][0].delta_x,
            data["basis"][1].delta_x,
            delta_x0_real,
            delta_x1_real,
            shape,
        )
    return {
        "basis": StackedBasis(
            FundamentalPositionBasis(
                np.array([delta_x0_real[0], delta_x0_real[1]]), shape[0]
            ),
            FundamentalPositionBasis(
                np.array([delta_x1_real[0], delta_x1_real[1]]), shape[1]
            ),
            data["basis"][2],
        ),
        "data": points.reshape(-1).astype(np.complex_),
    }


def interpolate_energy_grid_fourier_nickel(
    data: UnevenPotential3d[Any, Any, Any],
    delta_x0_real: tuple[float, float],
    delta_x1_real: tuple[float, float],
    shape: tuple[_L0Inv, _L1Inv, _L2Inv],
) -> Potential[
    StackedBasisLike[
        FundamentalPositionBasis3d[_L0Inv],
        FundamentalPositionBasis3d[_L1Inv],
        FundamentalPositionBasis3d[_L2Inv],
    ]
]:
    """
    Interpolate an energy grid using the fourier method.

    In the xy direction we ignore the initial lattice constants
    """
    xy_interpolation = interpolate_energy_grid_xy_fourier_nickel(
        data, delta_x0_real, delta_x1_real, (shape[0], shape[1])
    )
    interpolated = interpolate_points_along_axis_spline(
        xy_interpolation["data"].reshape((shape[0], shape[1], -1)),  # type: ignore[arg-type]
        xy_interpolation["basis"][2].z_points,
        shape[2],
        -1,
    )

    return {
        "basis": StackedBasis(
            FundamentalPositionBasis(
                np.array([*xy_interpolation["basis"][0].delta_x, 0]),
                xy_interpolation["basis"][0].n,
            ),
            FundamentalPositionBasis(
                np.array([*xy_interpolation["basis"][1].delta_x, 0]),
                xy_interpolation["basis"][1].n,
            ),
            FundamentalPositionBasis(
                np.array(
                    [0, 0, data["basis"][2].z_points[-1] - data["basis"][2].z_points[0]]
                ),
                shape[2],
            ),
        ),
        "data": interpolated.ravel().astype(np.complex_),
    }


def interpolate_nickel_potential(
    potential: UnevenPotential3d[Any, Any, Any], shape: tuple[_L0Inv, _L1Inv, _L2Inv]
) -> Potential[
    StackedBasisLike[
        FundamentalPositionBasis3d[_L0Inv],
        FundamentalPositionBasis3d[_L1Inv],
        FundamentalPositionBasis3d[_L2Inv],
    ]
]:
    raw_data = load_raw_potential_points()
    x_width = np.max(raw_data["x_points"]) - np.min(raw_data["x_points"])  # type: ignore[operator]

    delta_x0_real = (float(2 * x_width), 0)
    delta_x1_real = (
        float(0.5 * delta_x0_real[0]),
        float(np.sqrt(3) * delta_x0_real[0] / 2),
    )

    normalized = normalize_potential(potential)
    truncated = truncate_potential(normalized, cutoff=0.4e-18, n=1, offset=1e-20)
    data = interpolate_energy_grid_fourier_nickel(
        truncated, delta_x0_real, delta_x1_real, shape
    )
    return undo_truncate_potential(data, cutoff=0.4e-18, n=1, offset=1e-20)


def get_interpolated_potential(
    shape: tuple[_L0Inv, _L1Inv, _L2Inv]
) -> Potential[
    StackedBasisLike[
        FundamentalPositionBasis3d[_L0Inv],
        FundamentalPositionBasis3d[_L1Inv],
        FundamentalPositionBasis3d[_L2Inv],
    ]
]:
    potential = get_raw_potential_reciprocal_grid()
    return interpolate_nickel_potential(potential, shape)


def extrapolate_uneven_potential(
    potential: UnevenPotential3d[Any, Any, Any]
) -> UnevenPotential3d[Any, Any, Any]:
    util = BasisUtil(potential["basis"])

    old_z_points = potential["basis"][2].z_points
    old_potential = potential["data"].reshape(util.shape)

    z_range = old_z_points[-1] - old_z_points[0]
    z_points = cast(
        np.ndarray[Any, Any],
        np.linspace(
            old_z_points[0] - z_range,
            old_z_points[0] + 2 * z_range,
            4 * potential["basis"][2].n,
        ),
    )

    return {
        "basis": StackedBasis(
            potential["basis"][0],
            potential["basis"][1],
            UnevenPotential3dZBasis(z_points),
        ),
        "data": cast(
            np.ndarray[Any, Any],
            scipy.interpolate.interp1d(
                old_z_points,
                old_potential,
                axis=2,
                kind="slinear",
                fill_value="extrapolate",
            )(z_points),
        ).reshape(-1),
    }


def _interpolated_potential_cache(shape: tuple[_L0Inv, _L1Inv, _L2Inv]) -> Path:
    return get_data_path(
        f"potential/interpolated_potential_{shape[0]}_{shape[1]}_{shape[2]}.npy"
    )


@npy_cached(_interpolated_potential_cache, load_pickle=True)
def get_interpolated_extrapolated_potential(
    shape: tuple[_L0Inv, _L1Inv, _L2Inv],
) -> Potential[
    StackedBasisLike[
        FundamentalPositionBasis3d[_L0Inv],
        FundamentalPositionBasis3d[_L1Inv],
        FundamentalPositionBasis3d[_L2Inv],
    ]
]:
    potential = extrapolate_uneven_potential(get_raw_potential_reciprocal_grid())
    return interpolate_nickel_potential(potential, shape)


def get_interpolated_potential_john_grid() -> (
    Potential[
        StackedBasisLike[
            FundamentalPositionBasis3d[Any],
            FundamentalPositionBasis3d[Any],
            FundamentalPositionBasis3d[Any],
        ]
    ]
):
    potential = get_raw_potential_reciprocal_grid()
    normalized = normalize_potential(potential)
    truncated = truncate_potential(normalized, cutoff=0.4e-18, n=1, offset=1e-20)

    raw_data = load_raw_potential_points()
    x_width = np.max(raw_data["x_points"]) - np.min(raw_data["x_points"])  # type: ignore[operator]
    y_width = np.max(raw_data["y_points"]) - np.min(raw_data["y_points"])  # type: ignore[operator]

    delta_x0_real = (float(2 * x_width), 0)
    delta_x1_real = (0, float(3 * y_width))

    data = interpolate_energy_grid_fourier_nickel(
        truncated, delta_x0_real, delta_x1_real, (60, 60, 100)
    )
    return undo_truncate_potential(data, cutoff=0.4e-18, n=1, offset=1e-20)


def get_interpolated_potential_reciprocal_grid() -> (
    Potential[
        StackedBasisLike[
            FundamentalPositionBasis3d[int],
            FundamentalPositionBasis3d[int],
            FundamentalPositionBasis3d[int],
        ]
    ]
):
    potential = get_raw_potential_reciprocal_grid()
    normalized = normalize_potential(potential)
    truncated = truncate_potential(normalized, cutoff=0.4e-18, n=1, offset=1e-20)

    data = interpolate_uneven_potential(truncated, (48, 48, 100))
    return undo_truncate_potential(data, cutoff=0.4e-18, n=1, offset=1e-20)
