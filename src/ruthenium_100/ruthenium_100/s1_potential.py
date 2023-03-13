import numpy as np

from surface_potential_analysis.energy_data import EnergyPoints, load_energy_points

from .surface_data import get_data_path


def load_raw_data() -> EnergyPoints:
    path = get_data_path("raw_data.json")
    points = load_energy_points(path)
    max_point = np.max(points["points"])
    min_point = np.min(points["points"])
    points["points"][np.argmin(points["points"])] = 1.3 * max_point - 0.3 * min_point
    return points
