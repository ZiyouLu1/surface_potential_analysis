from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Generic, TypedDict, TypeVar

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

_L0Cov = TypeVar("_L0Cov", bound=int, covariant=True)


class PointPotential(TypedDict, Generic[_L0Cov]):
    """Represents an uneven potential, given at a series of points in the unit cell."""

    x_points: np.ndarray[tuple[_L0Cov], np.dtype[np.float_]]
    y_points: np.ndarray[tuple[_L0Cov], np.dtype[np.float_]]
    z_points: np.ndarray[tuple[_L0Cov], np.dtype[np.float_]]
    points: np.ndarray[tuple[_L0Cov], np.dtype[np.float_]]


def load_point_potential_json(
    path: Path,
) -> PointPotential[Any]:
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
