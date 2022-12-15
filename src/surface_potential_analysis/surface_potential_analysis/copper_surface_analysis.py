import os
from pathlib import Path
from typing import Tuple

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
    load_energy_eigenstates,
    save_energy_eigenstates,
)
from .energy_data.plot_energy_data import (
    plot_xz_plane_energy,
    plot_z_direction_energy_comparison,
    plot_z_direction_energy_data,
)
from .energy_data.plot_energy_eigenstates import (
    plot_eigenstate_positions,
    plot_lowest_band_in_kx,
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
    fig.savefig(path)


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
    path = get_data_path("copper_eigenstates_grid_3.json")

    generate_energy_eigenstates_grid(path, h, grid_size=8)


def plot_wavepacket_positions():
    path = get_data_path("copper_eigenstates_grid_faked.json")
    eigenstates = load_energy_eigenstates(path)

    fig, _, _ = plot_eigenstate_positions(eigenstates)
    fig.show()
    save_figure(fig, "faked_eigenstates_grid_points.png")


def plot_wavepacket_2D():
    # path = get_data_path("copper_eigenstates_grid_3.json")
    # eigenstates = load_energy_eigenstates(path)

    # h = generate_hamiltonian(eigenstates["resolution"])
    # normalized = normalize_eigenstate_phase(h, eigenstates)
    path = get_data_path("copper_eigenstates_grid_normalized.json")
    # save_energy_eigenstates(normalized, path)
    normalized = load_energy_eigenstates(path)
    h = generate_hamiltonian(normalized["resolution"])

    fig, _, _ = plot_wavepacket_in_xy(h, normalized)
    fig.show()
    save_figure(fig, "wavepacket3_eigenstates_2D.png")


def compare_wavefunction_2D():
    path = get_data_path("copper_eigenstates_grid_normalized.json")
    eigenstates = load_energy_eigenstates(path)

    config = eigenstates["eigenstate_config"]
    eigenvectors = eigenstates["eigenvectors"]
    h = SurfaceHamiltonian(
        eigenstates["resolution"],
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


def test_wavefunction_similarity():
    path = get_data_path("copper_eigenstates_grid_normalized.json")
    eigenstates = load_energy_eigenstates(path)

    config = eigenstates["eigenstate_config"]
    h = SurfaceHamiltonian(
        eigenstates["resolution"],
        potential={"dz": 0, "points": [[[]]]},
        config=config,
        potential_offset=0,
    )

    x_points = np.linspace(0, config["delta_x"], 100)
    points = np.array(
        [x_points, np.zeros_like(x_points), config["delta_x"] * np.ones_like(x_points)]
    ).T

    eigenvector1 = eigenstates["eigenvectors"][0]
    wavefunction_1 = h.calculate_wavefunction(points, eigenvector1)

    eigenvector2 = eigenstates["eigenvectors"][144]
    wavefunction_2 = h.calculate_wavefunction(points, eigenvector2)

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
