"""
Represents a potential in a given basis.

The potential is an operator, diagonal in position basis, and therefor can be stored as l
rather than l**2 points
"""

from __future__ import annotations

from .point_potential import PointPotential3d, load_point_potential_json
from .potential import (
    UnevenPotential3d,
    interpolate_uneven_potential,
    load_potential,
    load_uneven_potential,
    normalize_potential,
    truncate_potential,
    undo_truncate_potential,
)

__all__ = [
    # point potential
    "PointPotential3d",
    "load_point_potential_json",
    # potential
    "UnevenPotential3d",
    "interpolate_uneven_potential",
    "load_potential",
    "load_uneven_potential",
    "normalize_potential",
    "truncate_potential",
    "undo_truncate_potential",
]
