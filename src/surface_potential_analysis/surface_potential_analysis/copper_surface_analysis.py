import os
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .energy_data.energy_data import (
    as_interpolation,
    fill_surface_from_z_maximum,
    get_xy_points_delta,
    interpolate_energies_grid,
    load_energy_data,
    normalize_energy,
    save_energy_data,
    truncate_energy,
)
from .energy_data.energy_eigenstates import (
    EnergyEigenstates,
    load_energy_eigenstates,
    load_wavepacket_grid,
    save_energy_eigenstates,
    save_wavepacket_grid,
)
from .energy_data.plot_energy_data import (
    plot_xz_plane_energy,
    plot_z_direction_energy_comparison,
    plot_z_direction_energy_data,
)
from .energy_data.plot_energy_eigenstates import (
    plot_eigenstate_positions,
    plot_lowest_band_in_kx,
    plot_wavepacket_grid_x,
    plot_wavepacket_grid_xy,
    plot_wavepacket_grid_xz,
)
from .energy_data.sho_config import (
    EigenstateConfig,
    generate_sho_config_minimum,
    plot_interpolation_with_sho,
)
from .hamiltonian import (
    SurfaceHamiltonian,
    calculate_eigenvalues,
    calculate_energy_eigenstates,
    calculate_wavepacket_grid,
    generate_energy_eigenstates_grid,
    normalize_eigenstate_phase,
)
from .plot_surface_hamiltonian import (
    plot_bands_occupation,
    plot_eigenvector_through_bridge,
    plot_eigenvector_z,
    plot_first_4_eigenvectors,
    plot_wavefunction_difference_in_xy,
    plot_wavefunction_in_xy,
    plot_wavepacket_in_xy,
)


def get_out_path(filename: str) -> Path:
    out_folder_env = os.getenv("OUT_FOLDER")
    out_folder = (
        Path(out_folder_env)
        if out_folder_env is not None
        else Path(__file__).parent.parent.parent / "out"
    )
    return out_folder / filename


def save_figure(fig: Figure, filename: str) -> None:
    path = get_out_path(filename)
    fig.savefig(path)  # type: ignore


def get_data_path(filename: str) -> Path:
    data_folder_env = os.getenv("DATA_FOLDER")
    data_folder = (
        Path(data_folder_env)
        if data_folder_env is not None
        else Path(__file__).parent.parent.parent / "data"
    )
    return data_folder / filename


def load_raw_copper_data():
    path = get_data_path("copper_raw_energies.json")
    return load_energy_data(path)


def load_interpolated_copper_data():
    path = get_data_path("copper_interpolated_energies.json")
    return load_energy_data(path)


# def load_interpolated_copper_data_hd():
#     path = Path(__file__).parent / "data" / "copper_interpolated_energies_hd.json"
#     return load_energy_data(path)


def load_nc_raw_copper_data():
    path = get_data_path("copper_nc_raw_energies.json")
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


def generate_sho_config():
    data = load_interpolated_copper_data()
    interpolation = as_interpolation(data)
    mass = 1.6735575e-27
    return generate_sho_config_minimum(interpolation, mass, initial_guess=1e14)


def generate_hamiltonian(resolution: Tuple[int, int, int] = (1, 1, 1)):
    data = load_interpolated_copper_data()
    interpolation = as_interpolation(data)
    config: EigenstateConfig = {
        "mass": 1.6735575e-27,
        "sho_omega": 117905964225836.06,
        "delta_x": get_xy_points_delta(data["x_points"]),
        "delta_y": get_xy_points_delta(data["y_points"]),
    }

    z_offset = -1.840551985155284e-10
    return SurfaceHamiltonian(resolution, interpolation, config, z_offset)


def plot_interpolation_with_sho_config():
    data = load_interpolated_copper_data()
    interpolation = as_interpolation(data)
    # 80% 99514067252307.23
    # 50% 117905964225836.06
    config: EigenstateConfig = {
        "mass": 1.6735575e-27,
        "sho_omega": 117905964225836.06,  # 1e14,
        "delta_x": get_xy_points_delta(data["x_points"]),
        "delta_y": get_xy_points_delta(data["y_points"]),
    }
    z_offset = -1.840551985155284e-10
    plot_interpolation_with_sho(interpolation, config, z_offset)


def plot_first_copper_bands():
    h = generate_hamiltonian(resolution=(12, 12, 10))
    fig = plot_first_4_eigenvectors(h)
    save_figure(fig, "copper_first_4_bands.png")
    fig.show()


def list_first_copper_band_energies():
    h = generate_hamiltonian(resolution=(12, 12, 15))
    e_vals, _ = calculate_eigenvalues(h, 0, 0)
    print(list(np.sort(e_vals)[:40]))


def plot_copper_raw_data():
    data = load_raw_copper_data()
    data = normalize_energy(data)

    fig, ax = plot_z_direction_energy_data(data)
    ax.set_ylim(bottom=0, top=1e-18)
    fig.show()
    save_figure(fig, "copper_raw_data_z_direction.png")

    plot_xz_plane_energy(data)


def plot_copper_nc_data():
    data = normalize_energy(load_nc_raw_copper_data())

    fig, ax = plot_z_direction_energy_data(data)
    ax.set_ylim(bottom=-0.1e-18, top=1e-18)
    fig.show()
    save_figure(fig, "copper_raw_data_z_direction_nc.png")


def plot_copper_interpolated_data():
    data = load_interpolated_copper_data()
    raw_data = normalize_energy(load_raw_copper_data())

    fig, ax = plot_z_direction_energy_comparison(data, raw_data)
    ax.set_ylim(bottom=0, top=1e-18)
    fig.show()
    save_figure(fig, "copper_interpolated_data_comparison.png")

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
    save_figure(fig, "copper_lowest_band.png")


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
    save_figure(fig, "copper_bands_occupation.png")


def generate_eigenstates_data():
    h1 = generate_hamiltonian(resolution=(14, 14, 10))

    kx_points = np.linspace(-h1.dkx / 2, h1.dkx / 2, 11)
    ky_points = np.zeros_like(kx_points)

    # eigenstates1 = calculate_energy_eigenstates(h1, kx_points, ky_points)
    # path = get_data_path( "copper_eigenstates_14_14_10.json")
    # save_energy_eigenstates(eigenstates1, path)

    # h2 = generate_hamiltonian(resolution=(12, 12, 10))
    # eigenstates2 = calculate_energy_eigenstates(h2, kx_points, ky_points)
    # path = get_data_path( "copper_eigenstates_12_12_10.json")
    # save_energy_eigenstates(eigenstates2, path)

    # h3 = generate_hamiltonian(resolution=(12, 12, 12))
    # eigenstates3 = calculate_energy_eigenstates(h3, kx_points, ky_points)
    # path = get_data_path( "copper_eigenstates_12_12_12.json")
    # save_energy_eigenstates(eigenstates3, path)

    # h = generate_hamiltonian(resolution=(12, 12, 14))
    # eigenstates = calculate_energy_eigenstates(h, kx_points, ky_points)
    # path = get_data_path( "copper_eigenstates_12_12_14.json")
    # save_energy_eigenstates(eigenstates, path)

    # h = generate_hamiltonian(resolution=(12, 12, 15))
    # eigenstates = calculate_energy_eigenstates(h, kx_points, ky_points)
    # path = get_data_path( "copper_eigenstates_12_12_15.json")
    # save_energy_eigenstates(eigenstates, path)

    # h6 = generate_hamiltonian(resolution=(13, 13, 15))
    # eigenstates6 = calculate_energy_eigenstates(h6, kx_points, ky_points)
    # path = get_data_path( "copper_eigenstates_12_12_16.json")
    # save_energy_eigenstates(eigenstates6, path)

    h = generate_hamiltonian(resolution=(10, 10, 15))
    eigenstates = calculate_energy_eigenstates(h, kx_points, ky_points)
    path = get_data_path("copper_eigenstates_10_10_15.json")
    save_energy_eigenstates(eigenstates, path)

    # h = generate_hamiltonian(resolution=(13, 13, 15))
    # eigenstates = calculate_energy_eigenstates(h, kx_points, ky_points)
    # path = get_data_path( "copper_eigenstates_13_13_15.json")
    # save_energy_eigenstates(eigenstates, path)


def analyze_eigenvalue_convergence():

    fig, ax = plt.subplots()

    path = get_data_path("copper_eigenstates_12_12_10.json")
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(12,12,10)")

    path = get_data_path("copper_eigenstates_12_12_12.json")
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(12,12,12)")

    path = get_data_path("copper_eigenstates_12_12_14.json")
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(12,12,14)")

    path = get_data_path("copper_eigenstates_12_12_15.json")
    eigenstates = load_energy_eigenstates(path)
    _, _, ln = plot_lowest_band_in_kx(eigenstates, ax=ax)
    ln.set_label("(12,12,15)")

    path = get_data_path("copper_eigenstates_10_10_15.json")
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
    save_figure(fig, "copper_lowest_band_convergence.png")


def analyze_eigenvector_convergence_z():

    fig, ax = plt.subplots()

    path = get_data_path("copper_eigenstates_10_10_15.json")
    eigenstates = load_energy_eigenstates(path)
    h = generate_hamiltonian(resolution=eigenstates["resolution"])
    _, _, ln = plot_eigenvector_z(h, eigenstates["eigenvectors"][0], ax=ax)
    ln.set_label("(10,10,15) kx=G/2")

    path = get_data_path("copper_eigenstates_12_12_14.json")
    eigenstates = load_energy_eigenstates(path)
    h2 = generate_hamiltonian(resolution=eigenstates["resolution"])
    _, _, l2 = plot_eigenvector_z(h2, eigenstates["eigenvectors"][0], ax=ax)
    l2.set_label("(12,12,14) kx=G/2")

    path = get_data_path("copper_eigenstates_12_12_15.json")
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
    save_figure(fig, "copper_wfn_convergence.png")


def analyze_eigenvector_convergence_through_bridge():

    path = get_data_path("copper_eigenstates_12_12_15.json")
    eigenstates = load_energy_eigenstates(path)

    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    path = get_data_path("copper_eigenstates_12_12_15.json")
    eigenstates = load_energy_eigenstates(path)
    h = generate_hamiltonian(resolution=eigenstates["resolution"])
    _, _, ln = plot_eigenvector_through_bridge(h, eigenstates["eigenvectors"][5], ax=ax)
    _, _, _ = plot_eigenvector_through_bridge(
        h, eigenstates["eigenvectors"][5], ax=ax2, view="angle"
    )
    ln.set_label("(12,12,15)")

    path = get_data_path("copper_eigenstates_12_12_14.json")
    eigenstates = load_energy_eigenstates(path)
    h = generate_hamiltonian(resolution=eigenstates["resolution"])
    _, _, ln = plot_eigenvector_through_bridge(h, eigenstates["eigenvectors"][5], ax=ax)
    _, _, _ = plot_eigenvector_through_bridge(
        h, eigenstates["eigenvectors"][5], ax=ax2, view="angle"
    )
    ln.set_label("(12,12,14)")

    path = get_data_path("copper_eigenstates_10_10_15.json")
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
    save_figure(fig, "copper_wfn_convergence_through_bridge.png")


def generate_eigenstates_grid():
    h = generate_hamiltonian(resolution=(12, 12, 15))
    path = get_data_path("copper_eigenstates_grid_4.json")

    generate_energy_eigenstates_grid(path, h, grid_size=8)


def generate_normalized_eigenvalues():
    path = get_data_path("copper_eigenstates_grid_3.json")
    eigenstates = load_energy_eigenstates(path)

    h = generate_hamiltonian(eigenstates["resolution"])
    normalized = normalize_eigenstate_phase(h, eigenstates)
    path = get_data_path("copper_eigenstates_grid_normalized2.json")
    save_energy_eigenstates(normalized, path)


def plot_wavepacket_2D():
    path = get_data_path("copper_eigenstates_grid_normalized.json")
    normalized = load_energy_eigenstates(path)
    h = generate_hamiltonian(normalized["resolution"])

    fig, _, _ = plot_wavepacket_in_xy(h, normalized)
    fig.show()
    save_figure(fig, "wavepacket3_eigenstates_2D.png")


def filter_eigenstates(
    eigenstates: EnergyEigenstates, kx_points: List[float], ky_points: List[float]
) -> EnergyEigenstates:
    removed = np.zeros_like(eigenstates["kx_points"], dtype=bool)
    for kx in kx_points:
        removed = np.logical_or(removed, np.equal(eigenstates["kx_points"], kx))
    for ky in ky_points:
        removed = np.logical_or(removed, np.equal(eigenstates["ky_points"], ky))

    filtered = np.logical_not(removed)
    return {
        "eigenstate_config": eigenstates["eigenstate_config"],
        "eigenvalues": np.array(eigenstates["eigenvalues"])[filtered].tolist(),
        "eigenvectors": np.array(eigenstates["eigenvectors"])[filtered].tolist(),
        "kx_points": np.array(eigenstates["kx_points"])[filtered].tolist(),
        "ky_points": np.array(eigenstates["ky_points"])[filtered].tolist(),
        "resolution": eigenstates["resolution"],
    }


# Remove the extra point we repeated when generating eigenstates
def fix_eigenstates(eigenstates: EnergyEigenstates) -> EnergyEigenstates:
    kx_point = np.max(eigenstates["kx_points"])
    ky_point = np.max(eigenstates["ky_points"])
    return filter_eigenstates(eigenstates, [kx_point], [ky_point])


def calculate_wavepacket():
    path = get_data_path("copper_eigenstates_grid_normalized2.json")
    eigenstates = load_energy_eigenstates(path)

    filtered = fix_eigenstates(eigenstates)
    wavepacket = calculate_wavepacket_grid(filtered)
    path = get_data_path("copper_eigenstates_wavepacket.json")
    save_wavepacket_grid(wavepacket, path)


def calculate_wavepacket_approx():
    path = get_data_path("copper_eigenstates_grid_normalized2.json")
    eigenstates = load_energy_eigenstates(path)
    fig, _, _ = plot_eigenstate_positions(eigenstates)
    fig.show()

    filtered = fix_eigenstates(eigenstates)
    fig, _, _ = plot_eigenstate_positions(filtered)
    fig.show()

    wavepacket = calculate_wavepacket_grid(filtered, cutoff=200)
    path = get_data_path("copper_eigenstates_wavepacket_approx2.json")
    save_wavepacket_grid(wavepacket, path)


def filter_eigenstates_4_point(eigenstates: EnergyEigenstates):
    kx_points = np.sort(np.unique(eigenstates["kx_points"]))[1::2].tolist()
    ky_points = np.sort(np.unique(eigenstates["ky_points"]))[1::2].tolist()
    return filter_eigenstates(eigenstates, kx_points, ky_points)


def calculate_wavepacket_4_points_approx():
    path = get_data_path("copper_eigenstates_grid_normalized2.json")
    eigenstates = load_energy_eigenstates(path)

    filtered = fix_eigenstates(filter_eigenstates_4_point(eigenstates))
    wavepacket = calculate_wavepacket_grid(filtered, cutoff=200)
    path = get_data_path("copper_eigenstates_wavepacket_4_point_approx.json")
    save_wavepacket_grid(wavepacket, path)


def calculate_wavepacket_4_points():
    path = get_data_path("copper_eigenstates_grid_normalized2.json")
    eigenstates = load_energy_eigenstates(path)

    filtered = fix_eigenstates(filter_eigenstates_4_point(eigenstates))
    wavepacket = calculate_wavepacket_grid(filtered)
    path = get_data_path("copper_eigenstates_wavepacket_4_point.json")
    save_wavepacket_grid(wavepacket, path)


def filter_eigenstates_1_point(eigenstates: EnergyEigenstates):
    kx_points = list(set(kx for kx in eigenstates["kx_points"] if kx != 0))
    ky_points = list(set(ky for ky in eigenstates["ky_points"] if ky != 0))
    return filter_eigenstates(eigenstates, kx_points, ky_points)


def calculate_wavepacket_one_point():
    path = get_data_path("copper_eigenstates_grid_normalized2.json")
    eigenstates = load_energy_eigenstates(path)

    filtered = filter_eigenstates_1_point(eigenstates)
    wavepacket = calculate_wavepacket_grid(filtered)

    xv, yv, zv = np.meshgrid(
        wavepacket["x_points"], wavepacket["y_points"], wavepacket["z_points"]
    )
    coords = np.array([xv.ravel(), yv.ravel(), zv.ravel()]).T

    points = np.array(wavepacket["points"]).ravel()
    delta_x = eigenstates["eigenstate_config"]["delta_x"]
    delta_y = eigenstates["eigenstate_config"]["delta_y"]
    points = (
        points
        * np.sinc((coords[:, 0] - (delta_x / 2)) / delta_x)
        * np.sinc((coords[:, 1] - (delta_y / 2)) / delta_y)
    )
    wavepacket["points"] = points.reshape(xv.shape).tolist()

    path = get_data_path("copper_eigenstates_wavepacket_1_point.json")
    save_wavepacket_grid(wavepacket, path)


def plot_localized_wavepacket_grid():
    path = get_data_path("copper_eigenstates_wavepacket_4_point.json")
    wavepacket = load_wavepacket_grid(path)

    print(wavepacket["z_points"])
    fig, _, img = plot_wavepacket_grid_xy(wavepacket, z_ind=10, measure="real")
    img.set_norm("symlog")
    fig.show()
    save_figure(fig, "copper_eigenstates_wavepacket_xy_approx.png")

    fig, _, img = plot_wavepacket_grid_xy(wavepacket, z_ind=9, measure="imag")
    img.set_norm("symlog")
    fig.show()
    save_figure(fig, "copper_eigenstates_wavepacket_xy_approx_imag.png")

    fig, _, img = plot_wavepacket_grid_xy(wavepacket, z_ind=9, measure="abs")
    img.set_norm("symlog")
    fig.show()
    save_figure(fig, "copper_eigenstates_wavepacket_xy_approx_log.png")


def compare_wavefunction_4_8_points():
    path = get_data_path("copper_eigenstates_wavepacket_approx2.json")
    wavepacket_8 = load_wavepacket_grid(path)

    path = get_data_path("copper_eigenstates_wavepacket.json")
    wavepacket_8_full = load_wavepacket_grid(path)

    path = get_data_path("copper_eigenstates_wavepacket_4_point_approx.json")
    wavepacket_4 = load_wavepacket_grid(path)

    path = get_data_path("copper_eigenstates_wavepacket_4_point.json")
    wavepacket_4_full = load_wavepacket_grid(path)

    path = get_data_path("copper_eigenstates_wavepacket_1_point.json")
    wavepacket_1 = load_wavepacket_grid(path)

    fig, ax = plt.subplots()
    _, _, l1 = plot_wavepacket_grid_x(wavepacket_4, y_ind=48, z_ind=10, ax=ax)
    l1.set_label("4 point grid approx")
    _, _, l2 = plot_wavepacket_grid_x(wavepacket_4_full, y_ind=48, z_ind=10, ax=ax)
    l2.set_label("4 point grid")
    _, _, l3 = plot_wavepacket_grid_x(wavepacket_8, y_ind=48, z_ind=10, ax=ax)
    l3.set_label("8 point grid approx")
    _, _, l4 = plot_wavepacket_grid_x(wavepacket_8_full, y_ind=48, z_ind=10, ax=ax)
    l4.set_label("8 point grid")
    _, _, l5 = plot_wavepacket_grid_x(wavepacket_1, y_ind=96, z_ind=10, ax=ax)
    l5.set_label("1 point grid")

    ax.legend()
    ax.set_yscale("symlog")
    ax.set_title("Log plot of the abs 4 and 8 point wavefunctions")
    fig.show()
    save_figure(fig, "wavefunction_4_8_points_abs_comparison.png")

    fig, ax = plt.subplots()
    _, _, l1 = plot_wavepacket_grid_x(
        wavepacket_4, y_ind=48, z_ind=10, ax=ax, measure="real"
    )
    l1.set_label("4 point grid approx")
    _, _, l2 = plot_wavepacket_grid_x(
        wavepacket_4_full, y_ind=48, z_ind=10, ax=ax, measure="real"
    )
    l2.set_label("4 point grid")
    _, _, l3 = plot_wavepacket_grid_x(
        wavepacket_8, y_ind=48, z_ind=10, ax=ax, measure="real"
    )
    l3.set_label("8 point grid approx")
    _, _, l4 = plot_wavepacket_grid_x(
        wavepacket_8_full, y_ind=48, z_ind=10, ax=ax, measure="real"
    )
    l4.set_label("8 point grid")
    _, _, l5 = plot_wavepacket_grid_x(
        wavepacket_1, y_ind=96, z_ind=10, ax=ax, measure="real"
    )
    l5.set_label("1 point grid")

    ax.legend()
    ax.set_yscale("symlog")
    ax.set_title("Log plot of the real part of the 4 and 8 point wavefunctions")
    fig.show()
    save_figure(fig, "wavefunction_4_8_points_real_comparison.png")

    fig, ax = plt.subplots()
    _, _, l1 = plot_wavepacket_grid_x(
        wavepacket_4_full, y_ind=48, z_ind=10, ax=ax, measure="imag"
    )
    l1.set_label("4 point grid")
    _, _, l2 = plot_wavepacket_grid_x(
        wavepacket_8_full, y_ind=48, z_ind=10, ax=ax, measure="imag"
    )
    l2.set_label("8 point grid")
    _, _, l3 = plot_wavepacket_grid_x(
        wavepacket_1, y_ind=96, z_ind=10, ax=ax, measure="imag"
    )
    l3.set_label("1 point grid")

    ax.legend()
    ax.set_yscale("linear")
    ax.set_title("Imaginary part of the 4 and 8 point wavefunctions")
    fig.show()
    save_figure(fig, "wavefunction_4_8_points_imag_comparison.png")
    input()


def plot_wavefunction_xz_bridge():
    path = get_data_path("copper_eigenstates_wavepacket_approx2.json")
    wavepacket_8 = load_wavepacket_grid(path)

    print(wavepacket_8["y_points"][32])
    fig, ax = plt.subplots()
    _, _, im = plot_wavepacket_grid_xz(wavepacket_8, y_ind=32, ax=ax, measure="real")
    im.set_norm("symlog")

    fig.show()
    input()


def compare_wavefunction_2D():
    path = get_data_path("copper_eigenstates_grid_normalized.json")
    eigenstates = load_energy_eigenstates(path)

    config = eigenstates["eigenstate_config"]
    eigenvectors = eigenstates["eigenvectors"]
    h = SurfaceHamiltonian(
        resolution=eigenstates["resolution"],
        potential={"dz": 0, "points": [[[]]]},
        config=config,
        potential_offset=0,
    )
    fig, axs = plt.subplots(2, 3)
    (_, ax, _) = plot_wavefunction_in_xy(h, eigenvectors[0], axs[0][0])
    ax.set_title("(-dkx/2, -dky/2) at z=0")
    (_, ax, _) = plot_wavefunction_in_xy(h, eigenvectors[144], axs[0][1])
    ax.set_title("(0,0) at z=0")
    (_, ax, _) = plot_wavefunction_in_xy(h, eigenvectors[8], axs[0][2])
    ax.set_title("(-dkx/2, 0) at z=0")

    y_point = config["delta_x"]
    (_, ax, _) = plot_wavefunction_in_xy(h, eigenvectors[0], axs[1][0], y_point)
    ax.set_title("(-dkx/2, -dky/2) at z=delta_x")
    (_, ax, _) = plot_wavefunction_in_xy(h, eigenvectors[144], axs[1][1], y_point)
    ax.set_title("(0,0) at z=delta_x")
    (_, ax, _) = plot_wavefunction_in_xy(h, eigenvectors[8], axs[1][2], y_point)
    ax.set_title("(-dkx/2, 0) at z=delta_x")

    fig.tight_layout()
    fig.suptitle("Plot of absolute value of the Bloch wavefunctions")
    save_figure(fig, "Center and middle wavefunctions 2D")
    fig.show()

    fig, axs = plt.subplots(1, 2)
    (_, ax, _) = plot_wavefunction_difference_in_xy(
        h, eigenvectors[0], eigenvectors[144], axs[0]
    )
    ax.set_title("(-dkx/2, -dky/2) vs (0,0)")
    (_, ax, _) = plot_wavefunction_difference_in_xy(
        h, eigenvectors[8], eigenvectors[144], axs[1]
    )
    ax.set_title("(-dkx/2, 0) vs (0,0)")

    fig.suptitle("Plot of difference in the absolute value of the Bloch wavefunctions")
    fig.show()
    fig.tight_layout()
    save_figure(fig, "Center wavefunction diff 2D")


def calculate_eigenstate_cross_product():
    path = get_data_path("copper_eigenstates_grid_normalized.json")
    eigenstates = load_energy_eigenstates(path)

    eigenvector1 = eigenstates["eigenvectors"][0]
    eigenvector2 = eigenstates["eigenvectors"][144]

    prod = np.multiply(eigenvector1, np.conjugate(eigenvector2))
    print(prod)
    norm = np.sum(prod)
    print(norm)  # 0.95548


def test_wavefunction_similarity():
    path = get_data_path("copper_eigenstates_grid_normalized.json")
    eigenstates = load_energy_eigenstates(path)

    config = eigenstates["eigenstate_config"]
    h = SurfaceHamiltonian(
        resolution=eigenstates["resolution"],
        potential={"dz": 0, "points": [[[]]]},
        config=config,
        potential_offset=0,
    )

    x_points = np.linspace(0, config["delta_x"], 100)
    points = np.array(
        [x_points, np.zeros_like(x_points), config["delta_x"] * np.ones_like(x_points)]
    ).T

    eigenvector1 = eigenstates["eigenvectors"][0]
    wavefunction_1 = h.calculate_wavefunction_slow(points, eigenvector1)

    eigenvector2 = eigenstates["eigenvectors"][144]
    wavefunction_2 = h.calculate_wavefunction_slow(points, eigenvector2)

    fig, ax = plt.subplots()
    print(x_points)
    ax.plot(x_points, np.abs(wavefunction_1))
    ax.plot(x_points, np.abs(wavefunction_2))

    save_figure(fig, "Center and middle wavefunctions in x")
    fig.show()
    input()
    # for (wfn_1, wfn_2, point) in zip(wavefunctions_1, wavefunctions_2, points):

    np.testing.assert_allclose(
        wavefunction_1, wavefunction_2, atol=0.1 * np.max(np.abs(wavefunction_2))
    )


def investigate_approximate_eigenstates():
    path = get_data_path("copper_eigenstates_grid_normalized.json")
    eigenstates = load_energy_eigenstates(path)

    eigenvector = eigenstates["eigenvectors"][144]
    print(eigenvector.__len__())

    prod = np.multiply(eigenvector, np.conjugate(eigenvector))
    print(np.sum(prod))

    sorted = np.argsort(np.square(eigenvector))[::-1]
    print(sorted)
    approx_eigenvector = np.array(eigenvector)[sorted[:200]]
    approx_prod = np.multiply(approx_eigenvector, np.conjugate(approx_eigenvector))
    # 0.999620
    print(np.sum(approx_prod))
    print(np.sort(np.square(eigenvector))[200])


def test_block_wavefunction_fixed_phase_similarity():
    path = get_data_path("copper_eigenstates_grid.json")
    eigenstates = load_energy_eigenstates(path)

    n_eigenvectors = len(eigenstates["eigenvectors"])
    resolution = eigenstates["resolution"]

    h = generate_hamiltonian(eigenstates["resolution"])
    normalized = normalize_eigenstate_phase(h, eigenstates)

    fixed_phase_eigenvectors = np.array(normalized["eigenvectors"])

    reshaped = fixed_phase_eigenvectors.reshape(
        (n_eigenvectors, 2 * resolution[0] + 1, 2 * resolution[1] + 1, resolution[2])
    )
    for i in range(len(eigenstates["eigenvectors"])):
        for j in range(i, len(eigenstates["eigenvectors"])):
            print(i, j)
            np.testing.assert_allclose(
                np.sum(np.sum(reshaped[i], axis=0), axis=0),
                np.sum(np.sum(reshaped[j], axis=0), axis=0),
            )


# def generate_wavepacket() -> None:
#     path = get_data_path("copper_eigenstates_grid_faked.json")
#     eigenstates = load_energy_eigenstates(path)

#     eigenvectors = eigenstates["eigenvectors"]

#     phases = get_wavepacket_phases(eigenvectors, eigenstates["resolution"])
#     phase_factor = np.real_if_close(np.exp(-1j * np.array(phases)))
#     fixed_phase_eigenvectors = eigenvectors * phase_factor[:, np.newaxis]
#     input()

#     wavepacket_eigenvectors = np.sum(fixed_phase_eigenvectors, axis=0) / len(
#         eigenvectors
#     )

#     wavepacket: SurfaceWavepacket = {
#         "eigenvectors": wavepacket_eigenvectors.tolist(),
#         "resolution": eigenstates["resolution"],
#     }
#     path = get_data_path("copper_surface_wavepacket_grid_faked.json")
#     save_surface_wavepacket(wavepacket, path)


# def fixup_energy_eigenstates():
#     path = get_data_path("copper_eigenstates_grid.json")
#     eigenstates = load_energy_eigenstates(path)

#     data = mirror_energy_eigenstates(eigenstates)

#     path = get_data_path("copper_eigenstates_grid_faked.json")
#     save_energy_eigenstates(data, path)


if __name__ == "__main__":
    # generate_eigenstates_grid()

    # fixup_energy_eigenstates()
    # test_block_wavefunction_fixed_phase_similarity()
    # generate_eigenstates_data()
    # generate_eigenstates_data()
    # analyze_eigenvalue_convergence()
    # analyze_eigenvector_convergence_z()
    # analyze_eigenvector_convergence_through_bridge()
    print("Done")
