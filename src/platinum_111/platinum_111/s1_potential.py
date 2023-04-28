from __future__ import annotations

from surface_potential_analysis.potential.point_potential import (
    PointPotential,
    load_point_potential_json,
)

from .surface_data import get_data_path


def load_raw_data() -> PointPotential:
    path = get_data_path("raw_data.json")
    return load_point_potential_json(path)
