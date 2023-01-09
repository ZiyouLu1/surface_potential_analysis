import matplotlib.pyplot as plt
import numpy as np

from ..energy_data.energy_eigenstate import (
    EigenstateConfigUtil,
    get_eigenstate_list,
    load_energy_eigenstates,
)
from ..energy_data.plot_eigenstate import (
    plot_eigenstate_3D,
    plot_eigenstate_in_xy,
    plot_wavefunction_difference_in_xy,
)
from ..energy_data.plot_energy_eigenstates import plot_eigenstate_positions
from ..energy_data.plot_wavepacket_grid import (
    plot_wavepacket_grid_x,
    plot_wavepacket_grid_xy,
    plot_wavepacket_grid_xz,
    plot_wavepacket_grid_y_2D,
    plot_wavepacket_grid_z_2D,
    plot_wavepacket_in_xy,
)
from ..energy_data.wavepacket_grid import load_wavepacket_grid, symmetrize_wavepacket
from .copper_surface_data import get_data_path, save_figure
from .copper_surface_wavepacket import normalize_eigenstate_phase


def plot_wavepacket_points():
    path = get_data_path("copper_eigenstates_grid_offset.json")
    eigenstates = load_energy_eigenstates(path)
    fig, _, _ = plot_eigenstate_positions(eigenstates)

    fig.show()

    eigenstate_list = get_eigenstate_list(eigenstates)
    fig, _, _ = plot_eigenstate_3D(
        eigenstates["eigenstate_config"], eigenstate_list[-1]
    )

    fig.show()
    input()


def plot_wavepacket_2D():
    path = get_data_path("copper_eigenstates_grid_normalized.json")
    normalized = load_energy_eigenstates(path)

    fig, _, _ = plot_wavepacket_in_xy(normalized)
    fig.show()
    save_figure(fig, "wavepacket3_eigenstates_2D.png")


def plot_localized_wavepacket_grid():
    path = get_data_path("copper_eigenstates_wavepacket.json")
    wavepacket = symmetrize_wavepacket(load_wavepacket_grid(path))

    print(wavepacket["z_points"])
    fig, _, img = plot_wavepacket_grid_xy(wavepacket, z_ind=10, measure="real")
    img.set_norm("symlog")  # type: ignore
    fig.show()
    save_figure(fig, "copper_eigenstates_wavepacket_xy_approx.png")

    fig, _, img = plot_wavepacket_grid_xy(wavepacket, z_ind=9, measure="imag")
    img.set_norm("symlog")  # type: ignore
    fig.show()
    save_figure(fig, "copper_eigenstates_wavepacket_xy_approx_imag.png")

    fig, _, img = plot_wavepacket_grid_xy(wavepacket, z_ind=9, measure="abs")
    img.set_norm("symlog")  # type: ignore
    fig.show()
    save_figure(fig, "copper_eigenstates_wavepacket_xy_approx_log.png")
    input()


def plot_wavefunction_3D():
    path = get_data_path("copper_eigenstates_wavepacket_1_point.json")
    path = get_data_path("copper_eigenstates_wavepacket.json")
    path = get_data_path("copper_eigenstates_wavepacket_offset.json")
    wavepacket = load_wavepacket_grid(path)
    wavepacket = symmetrize_wavepacket(wavepacket)

    fig, _, _ = plot_wavepacket_grid_z_2D(wavepacket)
    fig.show()
    input()
    fig, _, _ = plot_wavepacket_grid_y_2D(wavepacket)
    fig.show()
    input()


def compare_wavefunction_4_8_points():
    path = get_data_path("copper_eigenstates_wavepacket_offset.json")
    wavepacket_offset = load_wavepacket_grid(path)

    path = get_data_path("copper_eigenstates_wavepacket.json")
    wavepacket_8 = load_wavepacket_grid(path)

    path = get_data_path("copper_eigenstates_wavepacket_with_edge.json")
    wavepacket_edge = load_wavepacket_grid(path)

    path = get_data_path("copper_eigenstates_wavepacket_4_point.json")
    wavepacket_4 = load_wavepacket_grid(path)

    path = get_data_path("copper_eigenstates_wavepacket_1_point.json")
    wavepacket_1 = load_wavepacket_grid(path)

    fig, ax = plt.subplots()
    _, _, l1 = plot_wavepacket_grid_x(wavepacket_edge, y_ind=24, z_ind=10, ax=ax)
    l1.set_label("8 point grid edge")
    _, _, l2 = plot_wavepacket_grid_x(wavepacket_4, y_ind=48, z_ind=10, ax=ax)
    l2.set_label("4 point grid")
    _, _, l3 = plot_wavepacket_grid_x(wavepacket_offset, y_ind=48, z_ind=10, ax=ax)
    l3.set_label("4 point grid offset")
    _, _, l4 = plot_wavepacket_grid_x(wavepacket_8, y_ind=48, z_ind=10, ax=ax)
    l4.set_label("8 point grid")
    _, _, l5 = plot_wavepacket_grid_x(wavepacket_1, y_ind=48, z_ind=10, ax=ax)
    l5.set_label("1 point grid")

    ax.legend()
    ax.set_yscale("symlog")
    ax.set_title("Log plot of the abs 4 and 8 point wavefunctions")
    fig.show()
    save_figure(fig, "wavefunction_4_8_points_abs_comparison.png")

    fig, ax = plt.subplots()
    _, _, l1 = plot_wavepacket_grid_x(
        wavepacket_edge, y_ind=24, z_ind=10, ax=ax, measure="real"
    )
    l1.set_label("8 point grid edge")
    _, _, l2 = plot_wavepacket_grid_x(
        wavepacket_4, y_ind=48, z_ind=10, ax=ax, measure="real"
    )
    l2.set_label("4 point grid")
    _, _, l3 = plot_wavepacket_grid_x(
        wavepacket_offset, y_ind=48, z_ind=10, ax=ax, measure="real"
    )
    l3.set_label("4 point grid offset")
    _, _, l4 = plot_wavepacket_grid_x(
        wavepacket_8, y_ind=48, z_ind=10, ax=ax, measure="real"
    )
    l4.set_label("8 point grid")
    _, _, l5 = plot_wavepacket_grid_x(
        wavepacket_1, y_ind=48, z_ind=10, ax=ax, measure="real"
    )
    l5.set_label("1 point grid")

    ax.legend()
    ax.set_yscale("symlog")
    ax.set_title("Log plot of the real part of the 4 and 8 point wavefunctions")
    fig.show()
    save_figure(fig, "wavefunction_4_8_points_real_comparison.png")

    fig, ax = plt.subplots()
    _, _, l1 = plot_wavepacket_grid_x(
        wavepacket_4, y_ind=48, z_ind=10, ax=ax, measure="imag"
    )
    l1.set_label("4 point grid")
    _, _, l2 = plot_wavepacket_grid_x(
        wavepacket_8, y_ind=48, z_ind=10, ax=ax, measure="imag"
    )
    l2.set_label("8 point grid")
    _, _, l3 = plot_wavepacket_grid_x(
        wavepacket_1, y_ind=48, z_ind=10, ax=ax, measure="imag"
    )
    l3.set_label("1 point grid")

    ax.legend()
    ax.set_yscale("linear")
    ax.set_title("Imaginary part of the 4 and 8 point wavefunctions")
    fig.show()
    save_figure(fig, "wavefunction_4_8_points_imag_comparison.png")

    fig, ax = plt.subplots()
    _, _, l1 = plot_wavepacket_grid_x(
        wavepacket_4, y_ind=48, z_ind=12, ax=ax, measure="abs"
    )
    l1.set_label("4 point grid")
    _, _, l2 = plot_wavepacket_grid_x(
        wavepacket_8, y_ind=48, z_ind=12, ax=ax, measure="abs"
    )
    l2.set_label("8 point grid")
    _, _, l3 = plot_wavepacket_grid_x(
        wavepacket_1, y_ind=48, z_ind=12, ax=ax, measure="abs"
    )
    l3.set_label("1 point grid")
    ax.legend()
    ax.set_yscale("symlog")
    ax.set_title("Abs part of the 4 and 8 point wavefunctions")
    fig.show()
    save_figure(fig, "wavefunction_4_8_points_abs_comparison_max_height.png")
    input()


def plot_wavefunction_xz_bridge():
    path = get_data_path("copper_eigenstates_wavepacket_approx2.json")
    wavepacket_8 = load_wavepacket_grid(path)

    print(wavepacket_8["y_points"][32])
    fig, ax = plt.subplots()
    _, _, im = plot_wavepacket_grid_xz(wavepacket_8, y_ind=32, ax=ax, measure="real")
    im.set_norm("symlog")  # type: ignore

    fig.show()
    input()


def compare_wavefunction_2D():
    path = get_data_path("copper_eigenstates_grid_normalized.json")
    eigenstates = load_energy_eigenstates(path)

    config = eigenstates["eigenstate_config"]
    eigenstate_list = get_eigenstate_list(eigenstates)

    fig, axs = plt.subplots(2, 3)
    (_, ax, _) = plot_eigenstate_in_xy(config, eigenstate_list[0], axs[0][0])
    ax.set_title("(-dkx/2, -dky/2) at z=0")
    (_, ax, _) = plot_eigenstate_in_xy(config, eigenstate_list[144], axs[0][1])
    ax.set_title("(0,0) at z=0")
    (_, ax, _) = plot_eigenstate_in_xy(config, eigenstate_list[8], axs[0][2])
    ax.set_title("(-dkx/2, 0) at z=0")

    y_point = config["delta_x"]
    (_, ax, _) = plot_eigenstate_in_xy(config, eigenstate_list[0], axs[1][0], y_point)
    ax.set_title("(-dkx/2, -dky/2) at z=delta_x")
    (_, ax, _) = plot_eigenstate_in_xy(config, eigenstate_list[144], axs[1][1], y_point)
    ax.set_title("(0,0) at z=delta_x")
    (_, ax, _) = plot_eigenstate_in_xy(config, eigenstate_list[8], axs[1][2], y_point)
    ax.set_title("(-dkx/2, 0) at z=delta_x")

    fig.tight_layout()
    fig.suptitle("Plot of absolute value of the Bloch wavefunctions")
    save_figure(fig, "Center and middle wavefunctions 2D")
    fig.show()

    fig, axs = plt.subplots(1, 2)
    (_, ax, _) = plot_wavefunction_difference_in_xy(
        config, eigenstate_list[0], eigenstate_list[144], axs[0]
    )
    ax.set_title("(-dkx/2, -dky/2) vs (0,0)")
    (_, ax, _) = plot_wavefunction_difference_in_xy(
        config, eigenstate_list[8], eigenstate_list[144], axs[1]
    )
    ax.set_title("(-dkx/2, 0) vs (0,0)")

    fig.suptitle("Plot of difference in the absolute value of the Bloch wavefunctions")
    fig.show()
    fig.tight_layout()
    save_figure(fig, "Center wavefunction diff 2D")
    input()


def test_wavefunction_similarity() -> None:
    path = get_data_path("copper_eigenstates_grid_normalized.json")
    eigenstates = load_energy_eigenstates(path)

    config = eigenstates["eigenstate_config"]
    util = EigenstateConfigUtil(config)

    x_points = np.linspace(0, config["delta_x"], 100)
    points = np.array(
        [x_points, np.zeros_like(x_points), config["delta_x"] * np.ones_like(x_points)]
    ).T

    eigenstate_list = get_eigenstate_list(eigenstates)
    eigenstate1 = eigenstate_list[0]
    wavefunction_1 = util.calculate_wavefunction_slow(eigenstate1, points)

    eigenstate2 = eigenstate_list[144]
    wavefunction_2 = util.calculate_wavefunction_slow(eigenstate2, points)

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


# How different are the bloch wavefunctions
def calculate_eigenstate_cross_product() -> None:
    path = get_data_path("copper_eigenstates_grid_normalized.json")
    eigenstates = load_energy_eigenstates(path)

    eigenvector1 = eigenstates["eigenvectors"][0]
    eigenvector2 = eigenstates["eigenvectors"][144]

    prod = np.multiply(eigenvector1, np.conjugate(eigenvector2))
    print(prod)
    norm = np.sum(prod)
    print(norm)  # 0.95548


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
    resolution = eigenstates["eigenstate_config"]["resolution"]

    normalized = normalize_eigenstate_phase(eigenstates)

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
