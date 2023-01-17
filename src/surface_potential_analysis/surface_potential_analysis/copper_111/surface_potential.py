from ..energy_data.energy_data import (
    EnergyGrid,
    EnergyPoints,
    load_energy_grid,
    load_energy_points,
)
from .surface_data import get_data_path


def load_raw_data() -> EnergyPoints:
    path = get_data_path("raw_data.json")
    return load_energy_points(path)


def load_john_interpolation() -> EnergyGrid:
    path = get_data_path("john_interpolated_data.json")
    return load_energy_grid(path)
