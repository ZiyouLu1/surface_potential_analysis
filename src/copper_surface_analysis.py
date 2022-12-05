from datetime import datetime
from pathlib import Path
from typing import Tuple

from energy_data import (
    as_interpolation,
    fill_surface_from_z_maximum,
    interpolate_energies_grid,
    load_energy_data,
    normalize_energy,
    save_energy_data,
    truncate_energy,
)
from hamiltonian import SurfaceHamiltonian
from plot_energy_data import (
    plot_energy_eigenvalues,
    plot_ground_state,
    plot_interpolation_with_sho,
    plot_xz_plane_energy,
)
from sho_config import SHOConfig, generate_sho_config_minimum


def load_raw_copper_data():
    path = Path(__file__).parent / "data" / "copper_raw_energies.json"
    return load_energy_data(path)


def load_interpolated_copper_data():
    path = Path(__file__).parent / "data" / "copper_interpolated_energies_hd.json"
    return load_energy_data(path)


def load_clean_copper_data():
    data = load_raw_copper_data()
    data = normalize_energy(data)
    data = fill_surface_from_z_maximum(data)
    data = truncate_energy(data, cutoff=3e-18, n=6, offset=1e-20)
    return data


def generate_interpolated_copper_data():
    data = load_clean_copper_data()
    interpolated = interpolate_energies_grid(data, shape=(20, 20, 100))
    path = Path(__file__).parent / "data" / "copper_interpolated_energies.json"
    save_energy_data(interpolated, path)


def generate_sho_config():
    data = load_interpolated_copper_data()
    interpolation = as_interpolation(data)
    mass = 1.6735575e-27
    return generate_sho_config_minimum(interpolation, mass, initial_guess=1e14)


def generate_hamiltonian(resolution: Tuple[int, int, int] = (1, 1, 1)):
    data = load_interpolated_copper_data()
    interpolation = as_interpolation(data)
    config: SHOConfig = {
        "mass": 1.6735575e-27,
        "sho_omega": 117905964225836.06,
        "z_offset": -1.840551985155284e-10,
    }

    return SurfaceHamiltonian(resolution, interpolation, config)


def plot_interpolation_with_sho_config():
    data = load_interpolated_copper_data()
    interpolation = as_interpolation(data)
    # 80% 99514067252307.23
    # 50% 117905964225836.06
    config: SHOConfig = {
        "mass": 1.6735575e-27,
        "sho_omega": 117905964225836.06,  # 1e14,
        "z_offset": -1.840551985155284e-10,
    }

    plot_interpolation_with_sho(interpolation, config)


if __name__ == "__main__":
    # generate_interpolated_copper_data()
    # generate_hamiltonian()
    # generate_interpolated_copper_data()
    h = generate_hamiltonian(resolution=(24, 24, 10))
    # 153216
    # 311560
    # t1 = datetime.now()
    # print("start")
    # h._calculate_off_diagonal_energies_fast()
    # print((datetime.now() - t1).total_seconds())
    # print(h._calculate_off_diagonal_energies())
    plot_ground_state(h)
    # plot_energy_eigenvalues(h)

    input()
