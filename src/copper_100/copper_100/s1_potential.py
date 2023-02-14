from surface_potential_analysis.energy_data import (
    fill_surface_from_z_maximum,
    interpolate_energy_grid_3D_spline,
    interpolate_energy_grid_fourier,
    load_energy_grid,
    load_energy_grid_legacy,
    normalize_energy,
    save_energy_grid,
    truncate_energy,
)

from .surface_data import get_data_path


def load_raw_copper_data():
    path = get_data_path("copper_raw_energies.json")
    return load_energy_grid_legacy(path)


def load_relaxed_copper_data():
    path = get_data_path("raw_energies_relaxed_sp.json")
    return load_energy_grid_legacy(path)


def load_interpolated_relaxed_data():
    path = get_data_path("copper_interpolated_energies_relaxed.json")
    return load_energy_grid(path)


def load_spline_interpolated_relaxed_data():
    path = get_data_path("copper_spline_interpolated_energies_relaxed.json")
    return load_energy_grid(path)


def load_interpolated_copper_data():
    path = get_data_path("copper_interpolated_energies.json")
    return load_energy_grid_legacy(path)


def load_nc_raw_copper_data():
    path = get_data_path("copper_nc_raw_energies.json")
    return load_energy_grid_legacy(path)


def load_9h_copper_data():
    path = get_data_path("copper_9h_raw_energies.json")
    return load_energy_grid_legacy(path)


def load_simple_copper_data():
    path = get_data_path("copper_simple_raw_energies2.json")
    return load_energy_grid_legacy(path)


def load_clean_copper_data():
    data = load_raw_copper_data()
    data = normalize_energy(data)
    data = fill_surface_from_z_maximum(data)
    data = truncate_energy(data, cutoff=3e-18, n=6, offset=1e-20)
    return data


def generate_interpolated_copper_data_fourier():
    data = load_relaxed_copper_data()
    normalized = normalize_energy(data)
    interpolated = interpolate_energy_grid_fourier(normalized, shape=(60, 60, 120))
    path = get_data_path("copper_interpolated_energies_relaxed.json")
    save_energy_grid(interpolated, path)


def generate_interpolated_copper_data_3D_spline():
    data = load_relaxed_copper_data()
    normalized = normalize_energy(data)
    interpolated = interpolate_energy_grid_3D_spline(normalized, shape=(60, 60, 120))
    path = get_data_path("copper_spline_interpolated_energies_relaxed.json")
    save_energy_grid(interpolated, path)
