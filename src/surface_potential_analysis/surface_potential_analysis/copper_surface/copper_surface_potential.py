from ..energy_data.energy_data import (
    fill_surface_from_z_maximum,
    interpolate_energies_grid,
    load_energy_data,
    normalize_energy,
    save_energy_data,
    truncate_energy,
)
from .copper_surface_data import get_data_path


def load_raw_copper_data():
    path = get_data_path("copper_raw_energies.json")
    return load_energy_data(path)


def load_interpolated_copper_data():
    path = get_data_path("copper_interpolated_energies.json")
    return load_energy_data(path)


def load_nc_raw_copper_data():
    path = get_data_path("copper_nc_raw_energies.json")
    return load_energy_data(path)


def load_9h_copper_data():
    path = get_data_path("copper_9h_raw_energies.json")
    return load_energy_data(path)


def load_clean_copper_data():
    data = load_raw_copper_data()
    data = normalize_energy(data)
    data = fill_surface_from_z_maximum(data)
    data = truncate_energy(data, cutoff=3e-18, n=6, offset=1e-20)
    return data


def generate_interpolated_copper_data():
    data = load_clean_copper_data()
    interpolated = interpolate_energies_grid(data, shape=(60, 60, 120))
    path = get_data_path("copper_interpolated_energies.json")
    save_energy_data(interpolated, path)
