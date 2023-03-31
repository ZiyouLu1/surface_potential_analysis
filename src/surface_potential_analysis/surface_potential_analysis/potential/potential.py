from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Generic, Tuple, TypedDict, TypeVar

import numpy as np

from surface_potential_analysis.basis import BasisUtil, PositionBasis
from surface_potential_analysis.basis_config import PositionBasisConfig
from surface_potential_analysis.interpolation import (
    interpolate_points_along_axis_spline,
    interpolate_points_rfftn,
)

_L0Cov = TypeVar("_L0Cov", bound=int, covariant=True)
_L1Cov = TypeVar("_L1Cov", bound=int, covariant=True)
_L2Cov = TypeVar("_L2Cov", bound=int, covariant=True)

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)

PotentialPoints = np.ndarray[tuple[_L0Inv, _L1Inv, _L2Inv], np.dtype[np.float_]]


class Potential(TypedDict, Generic[_L0Cov, _L1Cov, _L2Cov]):
    basis: PositionBasisConfig[_L0Cov, _L1Cov, _L2Cov]
    points: PotentialPoints[_L0Cov, _L1Cov, _L2Cov]


def save_potential(path: Path, potential: Potential[Any, Any, Any]) -> None:
    np.save(path, potential)


def load_potential(path: Path) -> Potential[Any, Any, Any]:
    return np.load(path, allow_pickle=True)[()]  # type:ignore


def load_potential_grid_json(path: Path) -> Potential[Any, Any, Any]:
    class SurfacePotentialRaw(TypedDict):
        delta_x0: list[int]
        delta_x1: list[int]
        delta_x2: list[int]
        points: list[list[list[float]]]

    with path.open("r") as f:
        out: SurfacePotentialRaw = json.load(f)
        points = np.array(out["points"])
        return {
            "basis": (
                {
                    "_type": "position",
                    "n": points.shape[0],
                    "delta_x": np.array(out["delta_x0"]),
                },
                {
                    "_type": "position",
                    "n": points.shape[1],
                    "delta_x": np.array(out["delta_x1"]),
                },
                {
                    "_type": "position",
                    "n": points.shape[2],
                    "delta_x": np.array(out["delta_x2"]),
                },
            ),
            "points": np.array(out["points"]),
        }


class UnevenPotential(TypedDict, Generic[_L0Cov, _L1Cov, _L2Cov]):
    basis: Tuple[
        PositionBasis[_L0Cov],
        PositionBasis[_L1Cov],
        np.ndarray[tuple[_L2Cov], np.dtype[np.float_]],
    ]
    points: PotentialPoints[_L0Cov, _L1Cov, _L2Cov]


def save_uneven_potential(
    path: Path, potential: UnevenPotential[Any, Any, Any]
) -> None:
    np.save(path, potential)


def load_uneven_potential(path: Path) -> UnevenPotential[Any, Any, Any]:
    return np.load(path, allow_pickle=True)[()]  # type:ignore


def load_uneven_potential_json(
    path: Path,
) -> UnevenPotential[Any, Any, Any]:
    class SurfacePotentialRaw(TypedDict):
        delta_x0: list[int]
        delta_x1: list[int]
        z_points: list[int]
        points: list[list[list[float]]]

    with path.open("r") as f:
        out: SurfacePotentialRaw = json.load(f)
        points = np.array(out["points"])

        return {
            "basis": (
                {
                    "_type": "position",
                    "n": points.shape[0],
                    "delta_x": np.array(out["delta_x0"]),
                },
                {
                    "_type": "position",
                    "n": points.shape[1],
                    "delta_x": np.array(out["delta_x1"]),
                },
                np.array(out["z_points"]),
            ),
            "points": points,
        }


GenericPotential = (
    Potential[_L0Inv, _L1Inv, _L2Inv] | UnevenPotential[_L0Inv, _L1Inv, _L2Inv]
)
_GPInv = TypeVar("_GPInv", bound=GenericPotential[Any, Any, Any])


def normalize_potential(data: _GPInv) -> _GPInv:
    points = data["points"]
    normalized_points = points - points.min()
    return {"points": normalized_points, "basis": data["basis"]}  # type: ignore


def truncate_potential(
    data: _GPInv, *, cutoff: float = 2e-17, n: int = 1, offset: float = 2e-18
) -> _GPInv:
    """
    Reduce the maximum energy by taking the transformation

    :math:`cutoff * np.log(1 + ((E + offset) / cutoff) ** n) ** (1 / n) - offset`

    For :math:`E << Cutoff` the energy is left unchanged. This can be useful to
    prevent the energy interpolation process from producing rabid oscillations
    """
    points = (
        cutoff * np.log(1 + ((data["points"] + offset) / cutoff) ** n) ** (1 / n)
        - offset
    )
    return {"points": points, "basis": data["basis"]}  # type: ignore


def undo_truncate_potential(
    data: _GPInv, *, cutoff: float = 2e-17, n: int = 1, offset: float = 2e-18
) -> _GPInv:
    """
    The reverse of truncate_energy
    """
    points = (
        cutoff * (np.exp((data["points"] + offset) / cutoff) - 1) ** (1 / n) - offset
    )
    return {"points": points, "basis": data["basis"]}  # type: ignore


def get_projected_x2_points(
    data: GenericPotential[Any, Any, _L0Inv]
) -> np.ndarray[tuple[_L0Inv], np.dtype[np.float_]]:
    a = data["basis"][2]
    if isinstance(a, dict):
        util = BasisUtil(a)
        return np.linalg.norm(util.x_points, axis=0)  # type: ignore
    else:
        return a


def interpolate_uneven_potential(
    data: UnevenPotential[Any, Any, Any], shape: tuple[_L0Inv, _L1Inv, _L2Inv]
) -> Potential[_L0Inv, _L1Inv, _L2Inv]:
    """
    Interpolate an energy grid using the fourier method

    Makes use of a fourier transform to increase the number of points
    in the xy plane of the energy grid, and a cubic spline to interpolate in the z direction
    """
    # TODO: maybe along axis
    xy_interpolated = interpolate_points_rfftn(
        data["points"], s=(shape[0], shape[1]), axes=(0, 1)
    )
    interpolated = interpolate_points_along_axis_spline(
        xy_interpolated, data["basis"][2], shape[2], axis=2
    )
    return {
        "basis": (
            {
                "_type": "position",
                "n": shape[0],
                "delta_x": data["basis"][0]["delta_x"],
            },
            {
                "_type": "position",
                "n": shape[1],
                "delta_x": data["basis"][1]["delta_x"],
            },
            {
                "_type": "position",
                "n": shape[2],
                "delta_x": np.array([0, 0, data["basis"][2][-1] - data["basis"][2][0]]),
            },
        ),
        "points": interpolated,  # type: ignore
    }


def mock_even_potential(
    uneven: UnevenPotential[_L0Inv, _L1Inv, _L2Inv]
) -> Potential[_L0Inv, _L1Inv, _L2Inv]:
    return {
        "basis": (
            uneven["basis"][0],
            uneven["basis"][1],
            {
                "_type": "position",
                "delta_x": np.array([0, 0, 1], dtype=float),
                "n": len(uneven["basis"][2]),  # type: ignore
            },
        ),
        "points": uneven["points"],
    }
