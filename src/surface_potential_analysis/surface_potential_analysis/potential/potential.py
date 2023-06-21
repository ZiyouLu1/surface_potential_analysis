from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Generic, TypedDict, TypeVar, overload

import numpy as np

from surface_potential_analysis.axis.axis import (
    FundamentalPositionAxis3d,
)
from surface_potential_analysis.basis.basis import (
    Basis,
    Basis1d,
    Basis2d,
    Basis3d,
    FundamentalPositionBasis3d,
)
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.util.interpolation import (
    interpolate_points_along_axis_spline,
    interpolate_points_rfftn,
)

if TYPE_CHECKING:
    from pathlib import Path


_L0Cov = TypeVar("_L0Cov", bound=int, covariant=True)
_L1Cov = TypeVar("_L1Cov", bound=int, covariant=True)
_L2Cov = TypeVar("_L2Cov", bound=int, covariant=True)

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)


PotentialPoints = np.ndarray[tuple[int], np.dtype[np.complex_]]

_B0Cov = TypeVar("_B0Cov", bound=Basis[Any], covariant=True)
_B0Inv = TypeVar("_B0Inv", bound=Basis[Any])
_B1d0Cov = TypeVar("_B1d0Cov", bound=Basis1d[Any], covariant=True)
_B2d0Cov = TypeVar("_B2d0Cov", bound=Basis2d[Any, Any], covariant=True)
_B3d0Cov = TypeVar("_B3d0Cov", bound=Basis3d[Any, Any, Any], covariant=True)


class Potential(TypedDict, Generic[_B0Cov]):
    """Represents a potential in an evenly spaced grid of points."""

    basis: _B0Cov
    vector: PotentialPoints


Potential1d = Potential[_B1d0Cov]

Potential2d = Potential[_B2d0Cov]

Potential3d = Potential[_B3d0Cov]


FundamentalPositionBasisPotential3d = Potential3d[
    FundamentalPositionBasis3d[_L0Inv, _L1Inv, _L2Inv]
]


def save_potential(path: Path, potential: Potential[Any]) -> None:
    """
    Save a potential in the npy format.

    Parameters
    ----------
    path : Path
    potential : Potential[Any, Any, Any]
    """
    np.save(path, potential)


def load_potential(path: Path) -> Potential[Any]:
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


def load_potential_grid_json(
    path: Path,
) -> Potential3d[FundamentalPositionBasis3d[Any, Any, Any]]:
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
                FundamentalPositionAxis3d(np.array(out["delta_x0"]), points.shape[0]),
                FundamentalPositionAxis3d(np.array(out["delta_x1"]), points.shape[1]),
                FundamentalPositionAxis3d(np.array(out["delta_x2"]), points.shape[2]),
            ),
            "vector": np.array(out["points"]),
        }


class UnevenPotential3d(TypedDict, Generic[_L0Cov, _L1Cov, _L2Cov]):
    """Represents a potential unevenly spaced in the z direction."""

    basis: tuple[
        FundamentalPositionAxis3d[_L0Cov],
        FundamentalPositionAxis3d[_L1Cov],
        np.ndarray[tuple[_L2Cov], np.dtype[np.float_]],
    ]
    vector: PotentialPoints


def save_uneven_potential(
    path: Path, potential: UnevenPotential3d[Any, Any, Any]
) -> None:
    """
    Save an uneven potential in the npy format.

    Parameters
    ----------
    path : Path
    potential : UnevenPotential[Any, Any, Any]
    """
    np.save(path, potential)


def load_uneven_potential(path: Path) -> UnevenPotential3d[Any, Any, Any]:
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
) -> UnevenPotential3d[Any, Any, Any]:
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
        points: list[float]

    with path.open("r") as f:
        out: SurfacePotentialRaw = json.load(f)
        points = np.array(out["points"])

        return {
            "basis": (
                FundamentalPositionAxis3d(np.array(out["delta_x0"]), points.shape[0]),
                FundamentalPositionAxis3d(np.array(out["delta_x1"]), points.shape[1]),
                np.array(out["z_points"]),
            ),
            "vector": points,
        }


_GenericPotential = Potential[_B0Inv] | UnevenPotential3d[_L0Inv, _L1Inv, _L2Inv]
_GPInv = TypeVar("_GPInv", bound=_GenericPotential[Any, Any, Any, Any])


@overload
def normalize_potential(data: Potential[_B0Inv]) -> Potential[_B0Inv]:
    ...


@overload
def normalize_potential(
    data: UnevenPotential3d[_L0Inv, _L1Inv, _L2Inv]
) -> UnevenPotential3d[_L0Inv, _L1Inv, _L2Inv]:
    ...


def normalize_potential(
    data: _GenericPotential[_B0Inv, _L0Inv, _L1Inv, _L2Inv]
) -> _GenericPotential[_B0Inv, _L0Inv, _L1Inv, _L2Inv]:
    """
    Set the minimum of the potential to 0.

    Parameters
    ----------
    data : _GPInv

    Returns
    -------
    _GPInv
    """
    points = data["vector"]
    normalized_points = points - points.min()
    return {"vector": normalized_points, "basis": data["basis"]}  # type: ignore[return-value,misc]


@overload
def truncate_potential(
    data: Potential[_B0Inv],
    *,
    cutoff: float = 2e-17,
    n: int = 1,
    offset: float = 2e-18,
) -> Potential[_B0Inv]:
    ...


@overload
def truncate_potential(
    data: UnevenPotential3d[_L0Inv, _L1Inv, _L2Inv],
    *,
    cutoff: float = 2e-17,
    n: int = 1,
    offset: float = 2e-18,
) -> UnevenPotential3d[_L0Inv, _L1Inv, _L2Inv]:
    ...


def truncate_potential(
    data: _GenericPotential[_B0Inv, _L0Inv, _L1Inv, _L2Inv],
    *,
    cutoff: float = 2e-17,
    n: int = 1,
    offset: float = 2e-18,
) -> _GenericPotential[_B0Inv, _L0Inv, _L1Inv, _L2Inv]:
    """
    Reduce the maximum energy by taking the transformation.

    :math:`cutoff * np.log(1 + ((E + offset) / cutoff) ** n) ** (1 / n) - offset`

    For :math:`E << Cutoff` the energy is left unchanged. This can be useful to
    prevent the energy interpolation process from producing rabid oscillations
    """
    points = (
        cutoff * np.log(1 + ((data["vector"] + offset) / cutoff) ** n) ** (1 / n)
        - offset
    )
    return {"vector": points, "basis": data["basis"]}  # type: ignore[return-value,misc]


def undo_truncate_potential(
    data: _GPInv, *, cutoff: float = 2e-17, n: int = 1, offset: float = 2e-18
) -> _GPInv:
    """Reverses truncate_potential."""
    points = (
        cutoff * (np.exp((data["vector"] + offset) / cutoff) - 1) ** (1 / n) - offset
    )
    return {"vector": points, "basis": data["basis"]}  # type: ignore[return-value]


def interpolate_uneven_potential(
    data: UnevenPotential3d[int, int, int], shape: tuple[_L0Inv, _L1Inv, _L2Inv]
) -> Potential3d[FundamentalPositionBasis3d[_L0Inv, _L1Inv, _L2Inv]]:
    """
    Interpolate an energy grid using the fourier method.

    Makes use of a fourier transform to increase the number of points
    in the xy plane of the energy grid, and a cubic spline to interpolate in the z direction
    """
    util = BasisUtil((data["basis"][0], data["basis"][1]))
    # TODO: maybe along axis
    xy_interpolated = interpolate_points_rfftn(
        data["vector"].reshape(*util.shape, len(data["basis"][2])).astype(np.float_),
        s=(shape[0], shape[1]),
        axes=(0, 1),
    )
    interpolated = interpolate_points_along_axis_spline(
        xy_interpolated, data["basis"][2], shape[2], axis=2
    )
    return {
        "basis": (
            FundamentalPositionAxis3d(data["basis"][0].delta_x, shape[0]),
            FundamentalPositionAxis3d(data["basis"][1].delta_x, shape[1]),
            FundamentalPositionAxis3d(
                np.array([0, 0, data["basis"][2][-1] - data["basis"][2][0]]), shape[2]
            ),
        ),
        "vector": interpolated.reshape(-1),  # type: ignore[typeddict-item]
    }


def mock_even_potential(
    uneven: UnevenPotential3d[_L0Inv, _L1Inv, _L2Inv]
) -> Potential3d[FundamentalPositionBasis3d[_L0Inv, _L1Inv, _L2Inv]]:
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
        "vector": uneven["vector"].reshape(-1),
    }
