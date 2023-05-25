from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Generic, TypedDict, TypeVar, overload

import numpy as np

from surface_potential_analysis.basis.basis import FundamentalPositionBasis
from surface_potential_analysis.util.interpolation import (
    interpolate_points_along_axis_spline,
    interpolate_points_rfftn,
)

if TYPE_CHECKING:
    from pathlib import Path

    from surface_potential_analysis.basis_config.basis_config import (
        FundamentalPositionBasisConfig,
    )

_L0Cov = TypeVar("_L0Cov", bound=int, covariant=True)
_L1Cov = TypeVar("_L1Cov", bound=int, covariant=True)
_L2Cov = TypeVar("_L2Cov", bound=int, covariant=True)

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)

PotentialPoints = np.ndarray[tuple[_L0Inv, _L1Inv, _L2Inv], np.dtype[np.float_]]


class Potential(TypedDict, Generic[_L0Cov, _L1Cov, _L2Cov]):
    """Represents a potential in an evenly spaced grid of points."""

    basis: FundamentalPositionBasisConfig[_L0Cov, _L1Cov, _L2Cov]
    points: PotentialPoints[_L0Cov, _L1Cov, _L2Cov]


def save_potential(path: Path, potential: Potential[Any, Any, Any]) -> None:
    """
    Save a potential in the npy format.

    Parameters
    ----------
    path : Path
    potential : Potential[Any, Any, Any]
    """
    np.save(path, potential)


def load_potential(path: Path) -> Potential[Any, Any, Any]:
    """
    Load a potential from the npy format.

    Parameters
    ----------
    path : Path

    Returns
    -------
    Potential[Any, Any, Any]
        _description_
    """
    return np.load(path, allow_pickle=True)[()]  # type: ignore[no-any-return]


def load_potential_grid_json(path: Path) -> Potential[Any, Any, Any]:
    """
    Load a potential from the JSON format.

    Parameters
    ----------
    path : Path

    Returns
    -------
    Potential[Any, Any, Any]
    """

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
                FundamentalPositionBasis(np.array(out["delta_x0"]), points.shape[0]),
                FundamentalPositionBasis(np.array(out["delta_x1"]), points.shape[1]),
                FundamentalPositionBasis(np.array(out["delta_x2"]), points.shape[2]),
            ),
            "points": np.array(out["points"]),
        }


class UnevenPotential(TypedDict, Generic[_L0Cov, _L1Cov, _L2Cov]):
    """Represents a potential unevenly spaced in the z direction."""

    basis: tuple[
        FundamentalPositionBasis[_L0Cov],
        FundamentalPositionBasis[_L1Cov],
        np.ndarray[tuple[_L2Cov], np.dtype[np.float_]],
    ]
    points: PotentialPoints[_L0Cov, _L1Cov, _L2Cov]


def save_uneven_potential(
    path: Path, potential: UnevenPotential[Any, Any, Any]
) -> None:
    """
    Save an uneven potential in the npy format.

    Parameters
    ----------
    path : Path
    potential : UnevenPotential[Any, Any, Any]
    """
    np.save(path, potential)


def load_uneven_potential(path: Path) -> UnevenPotential[Any, Any, Any]:
    """
    Load an uneven potential saved in the npy format.

    Parameters
    ----------
    path : Path

    Returns
    -------
    UnevenPotential[Any, Any, Any]
    """
    return np.load(path, allow_pickle=True)[()]  # type: ignore[no-any-return]


def load_uneven_potential_json(
    path: Path,
) -> UnevenPotential[Any, Any, Any]:
    """
    Load an uneven potential saved in the JSON format.

    Parameters
    ----------
    path : Path

    Returns
    -------
    UnevenPotential[Any, Any, Any]
    """

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
                FundamentalPositionBasis(np.array(out["delta_x0"]), points.shape[0]),
                FundamentalPositionBasis(np.array(out["delta_x1"]), points.shape[1]),
                np.array(out["z_points"]),
            ),
            "points": points,
        }


_GenericPotential = (
    Potential[_L0Inv, _L1Inv, _L2Inv] | UnevenPotential[_L0Inv, _L1Inv, _L2Inv]
)
_GPInv = TypeVar("_GPInv", bound=_GenericPotential[Any, Any, Any])


@overload
def normalize_potential(
    data: Potential[_L0Inv, _L1Inv, _L2Inv]
) -> Potential[_L0Inv, _L1Inv, _L2Inv]:
    ...


@overload
def normalize_potential(
    data: UnevenPotential[_L0Inv, _L1Inv, _L2Inv]
) -> UnevenPotential[_L0Inv, _L1Inv, _L2Inv]:
    ...


def normalize_potential(
    data: _GenericPotential[_L0Inv, _L1Inv, _L2Inv]
) -> _GenericPotential[_L0Inv, _L1Inv, _L2Inv]:
    """
    Set the minimum of the potential to 0.

    Parameters
    ----------
    data : _GPInv

    Returns
    -------
    _GPInv
    """
    points = data["points"]
    normalized_points = points - points.min()
    return {"points": normalized_points, "basis": data["basis"]}  # type: ignore[return-value,misc]


@overload
def truncate_potential(
    data: Potential[_L0Inv, _L1Inv, _L2Inv],
    *,
    cutoff: float = 2e-17,
    n: int = 1,
    offset: float = 2e-18,
) -> Potential[_L0Inv, _L1Inv, _L2Inv]:
    ...


@overload
def truncate_potential(
    data: UnevenPotential[_L0Inv, _L1Inv, _L2Inv],
    *,
    cutoff: float = 2e-17,
    n: int = 1,
    offset: float = 2e-18,
) -> UnevenPotential[_L0Inv, _L1Inv, _L2Inv]:
    ...


def truncate_potential(
    data: _GenericPotential[_L0Inv, _L1Inv, _L2Inv],
    *,
    cutoff: float = 2e-17,
    n: int = 1,
    offset: float = 2e-18,
) -> _GenericPotential[_L0Inv, _L1Inv, _L2Inv]:
    """
    Reduce the maximum energy by taking the transformation.

    :math:`cutoff * np.log(1 + ((E + offset) / cutoff) ** n) ** (1 / n) - offset`

    For :math:`E << Cutoff` the energy is left unchanged. This can be useful to
    prevent the energy interpolation process from producing rabid oscillations
    """
    points = (
        cutoff * np.log(1 + ((data["points"] + offset) / cutoff) ** n) ** (1 / n)
        - offset
    )
    return {"points": points, "basis": data["basis"]}  # type: ignore[return-value,misc]


def undo_truncate_potential(
    data: _GPInv, *, cutoff: float = 2e-17, n: int = 1, offset: float = 2e-18
) -> _GPInv:
    """Reverses truncate_potential."""
    points = (
        cutoff * (np.exp((data["points"] + offset) / cutoff) - 1) ** (1 / n) - offset
    )
    return {"points": points, "basis": data["basis"]}  # type: ignore[return-value]


def interpolate_uneven_potential(
    data: UnevenPotential[int, int, int], shape: tuple[_L0Inv, _L1Inv, _L2Inv]
) -> Potential[_L0Inv, _L1Inv, _L2Inv]:
    """
    Interpolate an energy grid using the fourier method.

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
            FundamentalPositionBasis(data["basis"][0].delta_x, shape[0]),
            FundamentalPositionBasis(data["basis"][1].delta_x, shape[1]),
            FundamentalPositionBasis(
                np.array([0, 0, data["basis"][2][-1] - data["basis"][2][0]]), shape[2]
            ),
        ),
        "points": interpolated,  # type: ignore[typeddict-item]
    }


def mock_even_potential(
    uneven: UnevenPotential[_L0Inv, _L1Inv, _L2Inv]
) -> Potential[_L0Inv, _L1Inv, _L2Inv]:
    """
    Generate a fake even potential from an uneven potential.

    Parameters
    ----------
    uneven : UnevenPotential[_L0Inv, _L1Inv, _L2Inv]

    Returns
    -------
    Potential[_L0Inv, _L1Inv, _L2Inv]
    """
    return {
        "basis": (
            uneven["basis"][0],
            uneven["basis"][1],
            {
                "_type": "position",
                "delta_x": np.array([0, 0, 1], dtype=float),
                "n": len(uneven["basis"][2]),  # type: ignore[typeddict-item]
            },
        ),
        "points": uneven["points"],
    }
