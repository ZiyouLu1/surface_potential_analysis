from surface_potential_analysis.energy_data import EnergyPoints, load_energy_points

from .surface_data import get_data_path


def load_raw_data() -> EnergyPoints:
    path = get_data_path("raw_data.json")
    return load_energy_points(path)
