from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from energy_data import (
    SurfaceWavepacket,
    as_interpolation,
    fill_surface_from_z_maximum,
    interpolate_energies_grid,
    load_energy_data,
    load_energy_eigenstates,
    normalize_energy,
    save_energy_data,
    save_energy_eigenstates,
    truncate_energy,
)
from hamiltonian import (
    SurfaceHamiltonian,
    calculate_energy_eigenstates,
    generate_energy_eigenstates_grid,
)
from plot_energy_data import (
    plot_bands_occupation,
    plot_eigenstate_positions,
    plot_eigenvector_through_bridge,
    plot_eigenvector_z,
    plot_first_4_eigenvectors,
    plot_interpolation_with_sho,
    plot_lowest_band_in_kx,
    plot_xz_plane_energy,
    plot_z_direction_energy_comparison,
    plot_z_direction_energy_data,
)
from sho_config import SHOConfig, generate_sho_config_minimum


def load_raw_copper_data():
    path = Path(__file__).parent / "data" / "copper_raw_energies.json"
    return load_energy_data(path)


def load_interpolated_copper_data():
    path = Path(__file__).parent / "data" / "copper_interpolated_energies.json"
    return load_energy_data(path)


# def load_interpolated_copper_data_hd():
#     path = Path(__file__).parent / "data" / "copper_interpolated_energies_hd.json"
#     return load_energy_data(path)


def load_nc_raw_copper_data():
    path = Path(__file__).parent / "data" / "copper_nc_raw_energies.json"
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


def plot_first_copper_eigenstates():
    h = generate_hamiltonian(resolution=(12, 12, 10))
    fig = plot_first_4_eigenvectors(h)
    fig.savefig("copper_first_4_eigenstates.png")
    fig.show()


def plot_copper_raw_data():
    data = load_raw_copper_data()
    data = normalize_energy(data)

    fig, ax = plot_z_direction_energy_data(data)
    ax.set_ylim(bottom=0, top=1e-18)
    fig.show()
    fig.savefig("copper_raw_data_z_direction.png")

    plot_xz_plane_energy(data)


def plot_copper_nc_data():
    data = normalize_energy(load_nc_raw_copper_data())

    fig, ax = plot_z_direction_energy_data(data)
    ax.set_ylim(bottom=-0.1e-18, top=1e-18)
    fig.show()
    fig.savefig("copper_raw_data_z_direction_nc.png")


def plot_copper_interpolated_data():
    data = load_interpolated_copper_data()
    raw_data = normalize_energy(load_raw_copper_data())

    fig, ax = plot_z_direction_energy_comparison(data, raw_data)
    ax.set_ylim(bottom=0, top=1e-18)
    fig.show()
    fig.savefig("copper_interpolated_data_comparison.png")

    plot_xz_plane_energy(data)


def plot_copper_band_structure():
    h = generate_hamiltonian(resolution=(12, 12, 10))

    kx_points = np.linspace(-h.dkx / 2, h.dkx / 2, 21)
    ky_points = np.zeros_like(kx_points)
    eigenstates = calculate_energy_eigenstates(h, kx_points, ky_points)

    fig, ax, _ = plot_lowest_band_in_kx(eigenstates)
    ax.set_title("Plot of energy against k for the lowest band of Copper for Ky=0")
    ax.set_xlabel("K /$m^-1$")
    ax.set_ylabel("energy / J")
    fig.show()
    fig.savefig("copper_lowest_band.png")


def plot_copper_bands_occupation():
    h = generate_hamiltonian(resolution=(12, 12, 10))
    # Plot the eigenstate occupation. Need to think about there 'mu' is
    # i.e. we have more than one hydrogen adsorbed on the surface
    # And interaction between hydrogen would also ruin things
    fig, ax, _ = plot_bands_occupation(h, temperature=60)
    ax.set_xlabel("Eigenvalue Index")
    ax.set_ylabel("Occupation Probability")
    ax.set_title(
        "Plot of occupation probability of each band according to the Boltzmann distribution"
    )
    fig.show()
    fig.savefig("copper_bands_occupation.png")


def generate_eigenstates_data():
    h1 = generate_hamiltonian(resolution=(14, 14, 10))

    kx_points = np.linspace(-h1.dkx / 2, h1.dkx / 2, 11)
    ky_points = np.zeros_like(kx_points)

    # eigenstates1 = calculate_energy_eigenstates(h1, kx_points, ky_points)
    # path = Path(__file__).parent / "data" / "copper_eigenstates_14_14_10.json"
    # save_energy_eigenstates(eigenstates1, path)

    # h2 = generate_hamiltonian(resolution=(12, 12, 10))
    # eigenstates2 = calculate_energy_eigenstates(h2, kx_points, ky_points)
    # path = Path(__file__).parent / "data" / "copper_eigenstates_12_12_10.json"
    # save_energy_eigenstates(eigenstates2, path)

    # h3 = generate_hamiltonian(resolution=(12, 12, 12))
    # eigenstates3 = calculate_energy_eigenstates(h3, kx_points, ky_points)
    # path = Path(__file__).parent / "data" / "copper_eigenstates_12_12_12.json"
    # save_energy_eigenstates(eigenstates3, path)

    # h = generate_hamiltonian(resolution=(12, 12, 14))
    # eigenstates = calculate_energy_eigenstates(h, kx_points, ky_points)
    # path = Path(__file__).parent / "data" / "copper_eigenstates_12_12_14.json"
    # save_energy_eigenstates(eigenstates, path)

    # h = generate_hamiltonian(resolution=(12, 12, 15))
    # eigenstates = calculate_energy_eigenstates(h, kx_points, ky_points)
    # path = Path(__file__).parent / "data" / "copper_eigenstates_12_12_15.json"
    # save_energy_eigenstates(eigenstates, path)

    # h6 = generate_hamiltonian(resolution=(13, 13, 15))
    # eigenstates6 = calculate_energy_eigenstates(h6, kx_points, ky_points)
    # path = Path(__file__).parent / "data" / "copper_eigenstates_12_12_16.json"
    # save_energy_eigenstates(eigenstates6, path)

    h = generate_hamiltonian(resolution=(10, 10, 15))
    eigenstates = calculate_energy_eigenstates(h, kx_points, ky_points)
    path = Path(__file__).parent / "data" / "copper_eigenstates_10_10_15.json"
    save_energy_eigenstates(eigenstates, path)

    # h = generate_hamiltonian(resolution=(13, 13, 15))
    # eigenstates = calculate_energy_eigenstates(h, kx_points, ky_points)
    # path = Path(__file__).parent / "data" / "copper_eigenstates_13_13_15.json"
    # save_energy_eigenstates(eigenstates, path)


def analyze_eigenvalue_convergence():

    fig, ax = plt.subplots()
    # path = Path(__file__).parent / "data" / "copper_eigenstates_14_14_10.json"
    # eigenstates = load_energy_eigenstates(path)
    # _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    # ln.set_label("(14,14,10)")

    path = Path(__file__).parent / "data" / "copper_eigenstates_12_12_10.json"
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(12,12,10)")

    path = Path(__file__).parent / "data" / "copper_eigenstates_12_12_12.json"
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(12,12,12)")

    path = Path(__file__).parent / "data" / "copper_eigenstates_12_12_14.json"
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(12,12,14)")

    path = Path(__file__).parent / "data" / "copper_eigenstates_12_12_15.json"
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(12,12,15)")

    path = Path(__file__).parent / "data" / "copper_eigenstates_10_10_15.json"
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(10,10,15)")

    ax.set_title(
        "Plot of energy against k for the lowest band of Copper for $K_y=0$\n"
        "showing convergence to about 2x$10^{-30}$J "
    )
    ax.set_xlabel("K /$m^{-1}$")
    ax.set_ylabel("energy / J")
    ax.legend()

    fig.tight_layout()
    fig.show()
    fig.savefig("copper_lowest_band_convergence.png")


def analyze_eigenvector_convergence_z():

    fig, ax = plt.subplots()

    path = Path(__file__).parent / "data" / "copper_eigenstates_10_10_15.json"
    eigenstates = load_energy_eigenstates(path)
    h = generate_hamiltonian(resolution=eigenstates["resolution"])
    _, _, ln = plot_eigenvector_z(h, eigenstates["eigenvectors"][0], ax=ax)
    ln.set_label("(10,10,15) kx=G/2")

    path = Path(__file__).parent / "data" / "copper_eigenstates_12_12_14.json"
    eigenstates = load_energy_eigenstates(path)
    h2 = generate_hamiltonian(resolution=eigenstates["resolution"])
    _, _, l2 = plot_eigenvector_z(h2, eigenstates["eigenvectors"][0], ax=ax)
    l2.set_label("(12,12,14) kx=G/2")

    path = Path(__file__).parent / "data" / "copper_eigenstates_12_12_15.json"
    eigenstates = load_energy_eigenstates(path)
    h = generate_hamiltonian(resolution=eigenstates["resolution"])
    _, _, ln = plot_eigenvector_z(h, eigenstates["eigenvectors"][0], ax=ax)
    ln.set_label("(12,12,15) kx=G/2")

    ax.set_title(
        "Plot of energy against k for the lowest band of Copper for $K_y=0$\n"
        "showing convergence to about 2x$10^{-30}$J "
    )
    ax.legend()
    fig.show()
    fig.savefig("copper_wfn_convergence.png")


def analyze_eigenvector_convergence_through_bridge():

    path = Path(__file__).parent / "data" / "copper_eigenstates_12_12_15.json"
    eigenstates = load_energy_eigenstates(path)

    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    path = Path(__file__).parent / "data" / "copper_eigenstates_12_12_15.json"
    eigenstates = load_energy_eigenstates(path)
    h = generate_hamiltonian(resolution=eigenstates["resolution"])
    _, _, ln = plot_eigenvector_through_bridge(h, eigenstates["eigenvectors"][5], ax=ax)
    _, _, _ = plot_eigenvector_through_bridge(
        h, eigenstates["eigenvectors"][5], ax=ax2, view="angle"
    )
    ln.set_label("(12,12,15)")

    path = Path(__file__).parent / "data" / "copper_eigenstates_12_12_14.json"
    eigenstates = load_energy_eigenstates(path)
    h = generate_hamiltonian(resolution=eigenstates["resolution"])
    _, _, ln = plot_eigenvector_through_bridge(h, eigenstates["eigenvectors"][5], ax=ax)
    _, _, _ = plot_eigenvector_through_bridge(
        h, eigenstates["eigenvectors"][5], ax=ax2, view="angle"
    )
    ln.set_label("(12,12,14)")

    path = Path(__file__).parent / "data" / "copper_eigenstates_10_10_15.json"
    eigenstates = load_energy_eigenstates(path)
    h = generate_hamiltonian(resolution=eigenstates["resolution"])
    _, _, ln = plot_eigenvector_through_bridge(h, eigenstates["eigenvectors"][5], ax=ax)
    _, _, _ = plot_eigenvector_through_bridge(
        h, eigenstates["eigenvectors"][5], ax=ax2, view="angle"
    )
    ln.set_label("(10,10,15)")

    ax2.set_ylim(-np.pi, np.pi)
    ax.set_title(
        "Plot of energy against k for the lowest band of Copper for $K_y=0$\n"
        "showing convergence to about 2x$10^{-30}$J "
    )
    ax.legend()
    fig.show()
    fig.savefig("copper_wfn_convergence_through_bridge.png")


def generate_eigenstates_grid():
    h = generate_hamiltonian(resolution=(12, 12, 15))

    path = Path(__file__).parent / "data" / "copper_eigenstates_grid.json"

    eigenstates = generate_energy_eigenstates_grid(path, h, grid_size=5)


def get_wavepacket_phases(
    eigenvectors: List[List[float]], resolution: Tuple[int, int, int]
):
    h = generate_hamiltonian(resolution=resolution)
    origin_point = [h.delta_x / 2, h.delta_y / 2, 0]

    phases = []
    for vector in eigenvectors:
        point_at_origin = h.calculate_wavefunction([origin_point], vector)[0]
        phases.append(np.angle(point_at_origin))
    return phases


def generate_wavepacket() -> SurfaceWavepacket:
    path = Path(__file__).parent / "data" / "copper_eigenstates_grid.json"
    eigenstates = load_energy_eigenstates(path)

    plot_eigenstate_positions(eigenstates)

    eigenvectors = eigenstates["eigenvectors"]

    phases = get_wavepacket_phases(eigenvectors, eigenstates["resolution"])

    fixed_phase_eigenvectors = np.divide(eigenvectors, phases)

    wavepacket_eigenvectors = np.sum(fixed_phase_eigenvectors, axis=0) / len(
        eigenvectors
    )

    return {
        "eigenvectors": wavepacket_eigenvectors.tolist(),
        "resolution": eigenstates["resolution"],
    }


if __name__ == "__main__":
    generate_eigenstates_grid()
    # generate_eigenstates_data()
    # generate_eigenstates_data()
    # analyze_eigenvalue_convergence()
    # analyze_eigenvector_convergence_z()
    # analyze_eigenvector_convergence_through_bridge()
    print("Done")
