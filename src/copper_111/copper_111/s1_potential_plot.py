import numpy as np
from matplotlib import pyplot as plt

from surface_potential_analysis.energy_data import (
    EnergyGrid,
    get_energy_points_xy_locations,
    normalize_energy,
    truncate_energy,
)
from surface_potential_analysis.energy_data_plot import (
    animate_energy_grid_3D_in_x1z,
    animate_energy_grid_3D_in_xy,
    compare_energy_grid_to_all_raw_points,
    plot_all_energy_points_z,
    plot_energy_grid_points,
    plot_energy_point_locations_on_grid,
    plot_energy_points_location,
    plot_potential_minimum_along_path,
    plot_z_direction_energy_comparison_111,
    plot_z_direction_energy_data_111,
)
from surface_potential_analysis.sho_wavefunction_plot import plot_sho_wavefunctions

from .s1_potential import (
    load_interpolated_grid,
    load_john_interpolation,
    load_raw_data,
    load_raw_data_grid,
)
from .surface_data import save_figure


def plot_raw_data_points():
    data = load_raw_data()
    fig, ax, _ = plot_energy_points_location(data)

    locations = get_energy_points_xy_locations(data)
    e_min = []
    for (x, y) in locations:
        idx = np.argwhere(
            np.logical_and(
                np.array(data["x_points"]) == x,
                np.array(data["y_points"]) == y,
            )
        )
        e_min.append(np.min(np.array(data["points"])[idx]))

    amin = np.argsort(e_min)
    x_min = locations[amin[0]][0]
    y_min = locations[amin[0]][1]
    ax.text(x_min, y_min, "FCC (lowest E)")

    x_min = locations[amin[1]][0]
    y_min = locations[amin[1]][1]
    ax.text(x_min, y_min, "HCP (second lowest E)")

    for i in range(2, 9):
        x_min = locations[amin[i]][0]
        y_min = locations[amin[i]][1]
        ax.text(x_min, y_min, f"({i})")

    x_min = locations[amin[-1]][0]
    y_min = locations[amin[-1]][1]
    ax.text(x_min, y_min, "Top (largest E)")

    fig.show()
    save_figure(fig, "nickel_raw_points.png")

    fig, ax = plot_all_energy_points_z(data)
    ax.set_ylim(0, 3 * 10**-19)

    ax.legend()
    fig.show()
    save_figure(fig, "nickel_raw_points_z.png")

    input()


def plot_raw_energy_grid_points():
    grid = normalize_energy(load_raw_data_grid())

    fig, _, _ = plot_energy_grid_points(grid)
    fig.show()

    fig, ax, _ = plot_z_direction_energy_data_111(grid)
    # ax.set_ylim(0, 0.2e-18)
    fig.show()

    fig, _, _ani = animate_energy_grid_3D_in_xy(grid)
    fig.show()

    truncated = truncate_energy(grid, cutoff=2e-19, n=1, offset=1e-20)
    fig, ax = plot_z_direction_energy_comparison_111(truncated, grid)
    ax.set_ylim(0, 0.2e-18)
    fig.show()

    ft_points = np.abs(np.fft.ifft2(grid["points"], axes=(0, 1)))
    ft_points[0, 0] = 0
    ft_grid: EnergyGrid = {
        "delta_x0": grid["delta_x0"],
        "delta_x1": grid["delta_x1"],
        "points": ft_points.tolist(),
        "z_points": grid["z_points"],
    }
    fig, ax, _ani1 = animate_energy_grid_3D_in_xy(ft_grid)
    # TODO: delta is wrong, plot generic points and then factor out into ft.
    ax.set_title("Plot of the ft of the raw potential")
    fig.show()
    input()


def plot_interpolated_energy_grid_points():
    grid = load_interpolated_grid()

    fig, ax, _ = plot_energy_grid_points(grid)
    fig.show()

    raw = normalize_energy(load_raw_data_grid())
    fig, ax = plot_z_direction_energy_comparison_111(grid, raw)
    # ax.set_ylim(0, 0.2e-18)
    fig.show()

    fig, ax, _ani = animate_energy_grid_3D_in_xy(grid, clim_max=0.2e-18)
    z_energies = np.min(grid["points"], axis=2)
    xy_min = np.unravel_index(np.argmin(z_energies), z_energies.shape)
    x0_min = xy_min[0] / (1 + z_energies.shape[0])
    x1_min = xy_min[1] / (1 + z_energies.shape[1])
    (line,) = ax.plot(
        x0_min * grid["delta_x0"][0] + x1_min * grid["delta_x1"][0],
        x0_min * grid["delta_x0"][1] + x1_min * grid["delta_x1"][1],
    )
    line.set_marker("x")

    z_energies = np.min(raw["points"], axis=2)
    xy_min = np.unravel_index(np.argmin(z_energies), z_energies.shape)
    x0_min = xy_min[0] / (1 + z_energies.shape[0])
    x1_min = xy_min[1] / (1 + z_energies.shape[1])
    (line,) = ax.plot(
        x0_min * raw["delta_x0"][0] + x1_min * raw["delta_x1"][0],
        x0_min * raw["delta_x0"][1] + x1_min * raw["delta_x1"][1],
    )
    line.set_marker("x")

    fig.show()
    input()

    ft_points = np.abs(np.fft.ifft2(grid["points"], axes=(0, 1)))
    ft_points[0, 0] = 0
    ft_grid: EnergyGrid = {
        "delta_x0": grid["delta_x0"],
        "delta_x1": grid["delta_x1"],
        "points": ft_points.tolist(),
        "z_points": grid["z_points"],
    }
    fig, ax, _ani1 = animate_energy_grid_3D_in_xy(ft_grid)
    ax.set_title("Plot of the ft of the interpolated potential")
    fig.show()
    input()


def plot_john_interpolated_points():
    data = load_john_interpolation()

    fig, ax, _anim1 = animate_energy_grid_3D_in_xy(data)
    ax.set_title(
        "Plot of the interpolated Copper surface potential\n" "through the z plane"
    )
    fig.show()

    fig, ax, _anim2 = animate_energy_grid_3D_in_x1z(data)
    ax.set_title(
        "Plot of the interpolated Copper surface potential\n" "through the y plane"
    )

    fig.show()
    input()


def compare_john_interpolation():
    raw_points = load_raw_data()
    interpolation = load_john_interpolation()

    fig, ax, _anim2 = plot_energy_point_locations_on_grid(raw_points, interpolation)
    fig.show()

    fig, ax = compare_energy_grid_to_all_raw_points(raw_points, interpolation)
    ax.set_ylim(0, 3 * 10**-19)
    ax.set_title("Comparison between raw and interpolated potential for Nickel")
    fig.show()
    input()
    save_figure(fig, "raw_interpolation_comparison.png")


def calculate_raw_fcc_hcp_energy_jump():
    raw_points = load_raw_data()

    x_points = np.array(raw_points["x_points"])
    y_points = np.array(raw_points["y_points"])
    points = np.array(raw_points["points"])

    p1 = points[(x_points == np.max(x_points))]
    p2 = points[np.logical_and(x_points == 0, y_points == np.max(y_points))]
    p3 = points[np.logical_and(x_points == 0, y_points == 0)]

    print(np.min(p1))  # 0.0
    print(np.min(p2))  # 2.95E-21J
    print(np.min(p3))  # 9.67E-20J


def plot_interpolation_with_sho_wavefunctions():
    """
    Is is possible that the SHO wavefunctions lie outside the interpolated potential
    or have energies for which they can see the truncation process.

    Plotting them alongside the interpolation in the hZ direction will allow us to
    diagnose these issues
    """

    grid = load_interpolated_grid()
    fig, ax = plt.subplots()
    plot_z_direction_energy_data_111(grid, ax=ax)
    plot_sho_wavefunctions(
        grid["z_points"],
        sho_omega=179704637926161.6,
        mass=1.6735575e-27,
        first_n=16,
        ax=ax,
    )
    ax.set_ylim(0, 0.5e-18)
    fig.show()

    save_figure(fig, "sho_wavefunctions_alongside_potential.png")
    input()


def plot_potential_minimum_along_diagonal():

    fig, ax = plt.subplots()

    interpolation = load_interpolated_grid()
    path = [(x, x) for x in range(np.shape(interpolation["points"])[0])]
    _, _, _ = plot_potential_minimum_along_path(interpolation, path, ax=ax)
    fig.show()
    save_figure(fig, "classical_trajectory_comparison.png")

    input()


def plot_potential_minimum_along_edge():
    interpolation = load_interpolated_grid()
    fig, ax = plt.subplots()

    path = [
        (np.shape(interpolation["points"])[0] - (1 + x), x)
        for x in range(np.shape(interpolation["points"])[0])
    ]
    print(path)
    _, _, line = plot_potential_minimum_along_path(interpolation, path, ax=ax)
    line.set_label("diagonal")

    path = [(x, 0) for x in range(np.shape(interpolation["points"])[0])]
    _, _, line = plot_potential_minimum_along_path(interpolation, path, ax=ax)
    line.set_label("x1=0")

    path = [(0, y) for y in range(np.shape(interpolation["points"])[1])]
    _, _, line = plot_potential_minimum_along_path(interpolation, path, ax=ax)
    line.set_label("x0=0")

    ax.legend()
    fig.show()
    ax.set_title(
        "plot of the potential along the edge and off diagonal. All three should be identical"
    )
    input()
