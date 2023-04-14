import numpy as np
from surface_potential_analysis.potential.point_potential import (
    PointPotential,
    load_point_potential_json,
)

from .surface_data import get_data_path


def load_raw_data() -> PointPotential:
    path = get_data_path("raw_data.json")
    points = load_point_potential_json(path)
    max_point = np.max(points["points"])
    min_point = np.min(points["points"])
    points["points"][np.argmin(points["points"])] = 1.3 * max_point - 0.3 * min_point
    return points
