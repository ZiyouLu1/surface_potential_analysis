from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Generic, TypedDict, TypeVar

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

_L0_co = TypeVar("_L0_co", bound=int, covariant=True)


class PointPotential3d(TypedDict, Generic[_L0_co]):
    """Represents an uneven potential, given at a series of points in the unit cell."""

    x_points: np.ndarray[tuple[_L0_co], np.dtype[np.float64]]
    y_points: np.ndarray[tuple[_L0_co], np.dtype[np.float64]]
    z_points: np.ndarray[tuple[_L0_co], np.dtype[np.float64]]
    points: np.ndarray[tuple[_L0_co], np.dtype[np.float64]]


def load_point_potential_json(
    path: Path,
) -> PointPotential3d[Any]:
    """
    Load a point potential from a JSON format.

    Parameters
    ----------
    path : Path

    Returns
    -------
    PointPotential[Any]
    """

    class SurfacePotentialRaw(TypedDict):
        x_points: list[float]
        y_points: list[float]
        z_points: list[float]
        points: list[float]

    with path.open("r") as f:
        out: SurfacePotentialRaw = json.load(f)
        return {
            "x_points": np.array(out["x_points"]),
            "y_points": np.array(out["y_points"]),
            "z_points": np.array(out["z_points"]),
            "points": np.array(out["points"]),
        }
