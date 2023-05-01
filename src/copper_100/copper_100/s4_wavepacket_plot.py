from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal, TypedDict

import matplotlib.pyplot as plt
import numpy as np
from surface_potential_analysis.eigenstate.eigenstate import (
    Eigenstate,
)
from surface_potential_analysis.eigenstate.plot import (
    plot_eigenstate_difference_2d,
    plot_eigenstate_x0x1,
)
from surface_potential_analysis.wavepacket.plot import (
    animate_wavepacket_x0x1,
    animate_wavepacket_x2x0,
    plot_wavepacket_sample_frequencies,
    plot_wavepacket_x0x1,
)

from .s4_wavepacket import normalize_eigenstate_phase_copper
from .surface_data import get_data_path, save_figure

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes


def plot_wavepacket_points() -> None:
    path = get_data_path("eigenstates_grid_0.json")
    load_energy_eigenstates(path)
    fig, _, _ = plot_wavepacket_sample_frequencies(eigenstates)

    fig.show()

    input()


def plot_wavepacket_at_z_origin() -> None:
    path = get_data_path("eigenstates_grid_0.json")
    eigenstates = load_energy_eigenstates(path)
    normalized = normalize_eigenstate_phase_copper(eigenstates)
    grid = calculate_wavepacket_grid_fourier(normalized, [0.0], (-4, 4), (-4, 4))

    fig, ax, _ = plot_wavepacket_x0x1(grid, measure="abs")
    fig.show()
    ax.set_title("Plot of abs(wavefunction) for z=0")
    save_figure(fig, "wavepacket_grid_z_origin.png")

    fig, ax, _ = plot_wavepacket_x0x1(grid, measure="real")
    fig.show()
    ax.set_title("Plot of real(wavefunction) for z=0")
    save_figure(fig, "wavepacket_grid_z_origin_real.png")

    fig, ax, _ = plot_wavepacket_x0x1(grid, measure="imag")
    fig.show()
    ax.set_title("Plot of imag(wavefunction) for z=0")
    save_figure(fig, "wavepacket_grid_z_origin_imag.png")
    input()


def plot_wavepacket_3d() -> None:
    path = get_data_path("eigenstates_grid_0.json")
    path = get_data_path("eigenstates_grid_1.json")
    eigenstates = load_energy_eigenstates(path)
    normalized = normalize_eigenstate_phase_copper(eigenstates)
    util = EigenstateConfigUtil(eigenstates["eigenstate_config"])
    z_points = np.linspace(-2 * util.characteristic_z, 2 * util.characteristic_z)
    grid = calculate_wavepacket_grid_fourier(normalized, z_points, (-4, 4), (-4, 4))

    fig, _, _ = animate_wavepacket_grid_3d_in_xy(grid)
    fig.show()
    input()
    fig, _, _ = animate_wavepacket_grid_3d_in_x0z(grid)
    fig.show()
    input()


def plot_relaxed_wavefunction_3d():
    path = get_data_path("relaxed_eigenstates_wavepacket_low_res.json")
    path = get_data_path("relaxed_eigenstates_wavepacket.json")
    path = get_data_path("relaxed_eigenstates_hd_wavepacket.json")
    path = get_data_path("relaxed_eigenstates_hd_wavepacket_flat.json")
    wavepacket = load_wavepacket_grid(path)
    wavepacket = reflect_wavepacket_in_axis(wavepacket, axis=1)

    fig, _, _ = animate_wavepacket_x0x1(wavepacket, measure="real")
    fig.show()
    input()

    fig, _, _ = animate_wavepacket_x2x0(wavepacket)
    fig.show()
    input()


def plot_new_wavepacket_relaxed():
    path = get_data_path("relaxed_eigenstates_wavepacket_new.json")
    wavepacket = load_wavepacket_grid(path)

    fig, _, _ = animate_wavepacket_x0x1(wavepacket, measure="real")
    fig.show()
    input()


def plot_wavepacket_difference_3d() -> None:
    path = get_data_path("relaxed_eigenstates_wavepacket.json")
    wavepacket_low_res = load_wavepacket_grid(path)
    path = get_data_path("relaxed_eigenstates_hd_wavepacket.json")
    wavepacket_hd = load_wavepacket_grid(path)

    new_points = np.subtract(wavepacket_hd["points"], wavepacket_low_res["points"])
    wavepacket: WavepacketGrid = {**wavepacket_hd, "points": new_points.tolist()}

    print(np.max(np.abs(wavepacket_hd["points"])))

    print(np.max(np.abs(wavepacket_low_res["points"])))

    fig, _, _anim0 = animate_wavepacket_x0x1(wavepacket, measure="real")
    fig.show()

    fig, _, _anim1 = animate_ft_wavepacket_grid_3d_in_xy(wavepacket, measure="real")
    fig.show()

    new_points = np.mean(
        [wavepacket_hd["points"], wavepacket_low_res["points"]], axis=0
    )
    wavepacket_averaged: WavepacketGrid = {
        **wavepacket_hd,
        "points": new_points.tolist(),
    }

    fig, _, _anim2 = animate_wavepacket_x0x1(wavepacket_averaged, measure="real")
    fig.show()

    fig, _, _anim3 = animate_ft_wavepacket_grid_3d_in_xy(
        wavepacket_averaged, measure="real"
    )
    fig.show()

    fig, _, _anim4 = animate_ft_wavepacket_grid_3d_in_xy(wavepacket_hd, measure="real")
    fig.show()

    input()


def plot_ft_hd_wavepacket_at_origin() -> None:
    path = get_data_path("relaxed_eigenstates_hd_wavepacket_flat.json")
    wavepacket = load_wavepacket_grid(path)
    fig, _, _ = plot_wavepacket_x0x1(wavepacket, z_ind=1, measure="real")
    fig.show()
    fig, _, _ = plot_ft_wavepacket_grid_xy(wavepacket, z_ind=1, measure="real")

    ft_points = np.fft.ifft2(wavepacket["points"], axes=(0, 1))

    new_points = np.fft.fft2(ft_points, axes=(0, 1))
    new_wavepacket: WavepacketGrid = {**wavepacket, "points": new_points.tolist()}
    fig, _, _ = plot_wavepacket_x0x1(new_wavepacket, z_ind=1, measure="real")
    fig.show()

    ft_points = np.fft.ifft2(wavepacket["points"], axes=(0, 1))
    new_ft_points = np.zeros_like(ft_points)
    new_ft_points[:+12, :, :] = ft_points[:+12, :, :]
    new_ft_points[-12:, :, :] = ft_points[-12:, :, :]
    new_ft_points[:, -12:, :] = ft_points[:, -12:, :]
    new_ft_points[:, :+12, :] = ft_points[:, :+12, :]

    new_points = np.fft.fft2(new_ft_points, axes=(0, 1))
    fixed_wavepacket: WavepacketGrid = {
        "delta_x0": wavepacket["delta_x0"],
        "delta_x1": wavepacket["delta_x1"],
        "z_points": wavepacket["z_points"],
        "points": new_points.tolist(),
    }
    fig, _, _ = plot_wavepacket_grid_xy(fixed_wavepacket, z_ind=1, measure="real")
    fig.show()
    ft_surface = get_reciprocal_surface(wavepacket)
    fig, ax, mesh = plot_points_on_surface_xy(
        ft_surface, ft_points.tolist(), z_ind=1, measure="abs"
    )
    fig.colorbar(mesh, ax=ax)

    fig.show()
    input()


def load_wavepacket_grid_legacy(path: Path) -> WavepacketGrid:
    class WavepacketGridLegacy(TypedDict):
        x_points: list[float]
        y_points: list[float]
        z_points: list[float]
        points: list[list[list[complex]]]

    with path.open("r") as f:
        out = json.load(f)
        points = np.array(out["real_points"]) + 1j * np.array(out["imag_points"])
        out["points"] = points.tolist()

        out2: WavepacketGridLegacy = out

        return {
            "points": out2["points"],
            "delta_x0": (out2["x_points"][-1] - out2["x_points"][0], 0),
            "delta_x1": (0, out2["y_points"][-1] - out2["y_points"][0]),
            "z_points": out2["z_points"],
        }


def compare_wavefunction_4_8_points() -> None:
    path = get_data_path("copper_eigenstates_wavepacket_offset.json")
    wavepacket_offset = load_wavepacket_grid_legacy(path)

    path = get_data_path("copper_eigenstates_wavepacket.json")
    wavepacket_8 = load_wavepacket_grid_legacy(path)

    path = get_data_path("copper_eigenstates_wavepacket_5.json")
    wavepacket_4_larger_k = load_wavepacket_grid_legacy(path)

    path = get_data_path("copper_eigenstates_wavepacket_with_edge.json")
    wavepacket_edge = load_wavepacket_grid_legacy(path)

    path = get_data_path("copper_eigenstates_wavepacket_4_point.json")
    wavepacket_4 = load_wavepacket_grid_legacy(path)

    path = get_data_path("copper_eigenstates_wavepacket_flat_band.json")
    wavepacket_1 = load_wavepacket_grid_legacy(path)

    fig, ax = plt.subplots()
    _, _, l2 = plot_wavepacket_grid_x1(wavepacket_4, x2_ind=48, z_ind=10, ax=ax)
    l2.set_label("4 point grid")
    _, _, l3 = plot_wavepacket_grid_x1(wavepacket_offset, x2_ind=48, z_ind=10, ax=ax)
    l3.set_label("4 point grid offset")
    _, _, l4 = plot_wavepacket_grid_x1(wavepacket_8, x2_ind=48, z_ind=10, ax=ax)
    l4.set_label("8 point grid")
    _, _, l5 = plot_wavepacket_grid_x1(wavepacket_1, x2_ind=48, z_ind=10, ax=ax)
    l5.set_label("1 point grid")
    _, _, l6 = plot_wavepacket_grid_x1(
        wavepacket_4_larger_k, x2_ind=48, z_ind=10, ax=ax
    )
    l6.set_label("4 point grid, larger k")

    ax.legend()
    ax.set_yscale("symlog")
    ax.set_title("Log plot of the abs 4 and 8 point wavefunctions")
    fig.show()
    save_figure(fig, "wavefunction_4_8_points_abs_comparison.png")

    fig, ax = plt.subplots()
    _, _, l1 = plot_wavepacket_grid_x1(
        wavepacket_edge, x2_ind=24, z_ind=10, ax=ax, measure="real"
    )
    l1.set_label("8 point grid edge")
    _, _, l2 = plot_wavepacket_grid_x1(
        wavepacket_4, x2_ind=48, z_ind=10, ax=ax, measure="real"
    )
    l2.set_label("4 point grid")
    _, _, l3 = plot_wavepacket_grid_x1(
        wavepacket_offset, x2_ind=48, z_ind=10, ax=ax, measure="real"
    )
    l3.set_label("4 point grid offset")
    _, _, l4 = plot_wavepacket_grid_x1(
        wavepacket_8, x2_ind=48, z_ind=10, ax=ax, measure="real"
    )
    l4.set_label("8 point grid")
    _, _, l5 = plot_wavepacket_grid_x1(
        wavepacket_1, x2_ind=48, z_ind=10, ax=ax, measure="real"
    )
    l5.set_label("1 point grid")
    _, _, l6 = plot_wavepacket_grid_x1(
        wavepacket_4_larger_k, x2_ind=48, z_ind=10, ax=ax, measure="real"
    )
    l6.set_label("4 point grid, larger k")

    ax.legend()
    ax.set_yscale("symlog")
    ax.set_title("Log plot of the real part of the 4 and 8 point wavefunctions")
    fig.show()
    save_figure(fig, "wavefunction_4_8_points_real_comparison.png")

    fig, ax = plt.subplots()
    _, _, l1 = plot_wavepacket_grid_x1(
        wavepacket_4, x2_ind=48, z_ind=10, ax=ax, measure="imag"
    )
    l1.set_label("4 point grid")
    _, _, l2 = plot_wavepacket_grid_x1(
        wavepacket_8, x2_ind=48, z_ind=10, ax=ax, measure="imag"
    )
    l2.set_label("8 point grid")
    _, _, l3 = plot_wavepacket_grid_x1(
        wavepacket_1, x2_ind=48, z_ind=10, ax=ax, measure="imag"
    )
    l3.set_label("1 point grid")

    ax.legend()
    ax.set_yscale("linear")
    ax.set_title("Imaginary part of the 4 and 8 point wavefunctions")
    fig.show()
    save_figure(fig, "wavefunction_4_8_points_imag_comparison.png")

    fig, ax = plt.subplots()
    _, _, l1 = plot_wavepacket_grid_x1(
        wavepacket_4, x2_ind=48, z_ind=12, ax=ax, measure="abs"
    )
    l1.set_label("4 point grid")
    _, _, l2 = plot_wavepacket_grid_x1(
        wavepacket_8, x2_ind=48, z_ind=12, ax=ax, measure="abs"
    )
    l2.set_label("8 point grid")
    _, _, l3 = plot_wavepacket_grid_x1(
        wavepacket_1, x2_ind=48, z_ind=12, ax=ax, measure="abs"
    )
    l3.set_label("1 point grid")
    ax.legend()
    ax.set_yscale("symlog")
    ax.set_title("Abs part of the 4 and 8 point wavefunctions")
    fig.show()
    save_figure(fig, "wavefunction_4_8_points_abs_comparison_max_height.png")
    input()


def compare_wavefunction_2d()->None:
    path = get_data_path("eigenstates_grid_0.json")
    eigenstates = load_energy_eigenstates(path)
    normalize_eigenstate_phase_copper(eigenstates)

    config = eigenstates["eigenstate_config"]
    eigenstate_list = get_eigenstate_list(eigenstates)

    fig, axs = plt.subplots(2, 3)
    (_, ax, _) = plot_eigenstate_x0x1(config, eigenstate_list[0], ax=axs[0][0])
    ax.set_title("(-dkx/2, -dky/2) at z=0")
    (_, ax, _) = plot_eigenstate_x0x1(config, eigenstate_list[144], ax=axs[0][1])
    ax.set_title("(0,0) at z=0")
    (_, ax, _) = plot_eigenstate_x0x1(config, eigenstate_list[8], ax=axs[0][2])
    ax.set_title("(-dkx/2, 0) at z=0")

    z_point = config["delta_x1"][0]
    (_, ax, _) = plot_eigenstate_x0x1(
        config, eigenstate_list[0], ax=axs[1][0], z_point=z_point
    )
    ax.set_title("(-dkx/2, -dky/2) at z=delta_x")
    (_, ax, _) = plot_eigenstate_x0x1(
        config, eigenstate_list[144], ax=axs[1][1], z_point=z_point
    )
    ax.set_title("(0,0) at z=delta_x")
    (_, ax, _) = plot_eigenstate_x0x1(
        config, eigenstate_list[8], ax=axs[1][2], z_point=z_point
    )
    ax.set_title("(-dkx/2, 0) at z=delta_x")

    fig.tight_layout()
    fig.suptitle("Plot of absolute value of the Bloch wavefunctions")
    save_figure(fig, "Center and middle wavefunctions 2D")
    fig.show()

    fig, axs = plt.subplots(1, 2)
    (_, ax, _) = plot_eigenstate_difference_2d(z_axis=2,
        config, eigenstate_list[0], eigenstate_list[144], axs[0]
    )
    ax.set_title("(-dkx/2, -dky/2) vs (0,0)")
    (_, ax, _) = plot_eigenstate_difference_2d(z_axis=2,
        config, eigenstate_list[8], eigenstate_list[144], axs[1]
    )
    ax.set_title("(-dkx/2, 0) vs (0,0)")

    fig.suptitle("Plot of difference in the absolute value of the Bloch wavefunctions")
    fig.show()
    fig.tight_layout()
    save_figure(fig, "Center wavefunction diff 2D")
    input()


def test_wavefunction_similarity() -> None:
    path = get_data_path("eigenstates_grid_0.json")
    eigenstates = load_energy_eigenstates(path)
    normalize_eigenstate_phase_copper(eigenstates)

    config = eigenstates["eigenstate_config"]
    util = EigenstateConfigUtil(config)

    x_points = np.linspace(0, config["delta_x1"][0], 100)
    points = np.array(
        [
            x_points,
            np.zeros_like(x_points),
            config["delta_x1"][0] * np.ones_like(x_points),
        ]
    ).T

    eigenstate_list = get_eigenstate_list(eigenstates)
    eigenstate1 = eigenstate_list[0]
    wavefunction_1 = util.calculate_wavefunction_slow(eigenstate1, points)

    eigenstate2 = eigenstate_list[144]
    wavefunction_2 = util.calculate_wavefunction_slow(eigenstate2, points)

    fig, ax = plt.subplots()

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
    path = get_data_path("eigenstates_grid_0.json")
    eigenstates = load_energy_eigenstates(path)
    normalize_eigenstate_phase_copper(eigenstates)

    eigenvector1 = eigenstates["eigenvectors"][0]
    eigenvector2 = eigenstates["eigenvectors"][144]

    prod = np.multiply(eigenvector1, np.conjugate(eigenvector2))
    print(prod)
    norm = np.sum(prod)
    print(norm)  # 0.95548


def investigate_approximate_eigenstates():
    path = get_data_path("eigenstates_grid_0.json")
    eigenstates = load_energy_eigenstates(path)
    normalize_eigenstate_phase_copper(eigenstates)

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
    path = get_data_path("eigenstates_grid_0.json")
    eigenstates = load_energy_eigenstates(path)

    n_eigenvectors = len(eigenstates["eigenvectors"])
    resolution = eigenstates["eigenstate_config"]["resolution"]

    normalized = normalize_eigenstate_phase_copper(eigenstates)

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


def plot_eigenstate_difference_in_z(
    config: EigenstateConfig,
    eig1: Eigenstate,
    eig2: Eigenstate,
    measure: Literal["abs", "rel"] = "abs",
    ax: Axes | None = None,
):
    fig, ax1 = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    util = EigenstateConfigUtil(config)
    z_points = np.linspace(-util.characteristic_z * 2, util.characteristic_z * 2, 100)
    points = [[util.delta_x0[0] / 2, util.delta_x1[1] / 2, pz] for pz in z_points]

    wfn0 = util.calculate_wavefunction_fast(eig1, points)
    wfn3 = util.calculate_wavefunction_fast(eig2, points)
    (line,) = ax1.plot(z_points, np.abs(wfn0))
    line.set_label(f"({eig1['kx']:.3E}, {eig1['ky']:.3E})")
    (line,) = ax1.plot(z_points, np.abs(wfn3))
    line.set_label(f"({eig2['kx']:.3E}, {eig2['ky']:.3E})")

    ax2 = ax1.twinx()
    if measure == "abs":
        (line,) = ax2.plot(z_points, np.abs(wfn0 - wfn3))
    else:
        (line,) = ax2.plot(z_points, np.abs(wfn0 - wfn3) / np.abs(wfn0))
    line.set_label("difference")
    line.set_linestyle("dashed")

    ax1.legend()
    return fig, ax2, line


def analyze_wavepacket_grid_1_points():
    path = get_data_path("eigenstates_grid_1.json")
    eigenstates = load_energy_eigenstates(path)
    filtered = filter_eigenstates_n_point(eigenstates, n=1)

    normalized = normalize_eigenstate_phase_copper(eigenstates)
    filtered_normalized = filter_eigenstates_n_point(normalized, n=1)

    util = EigenstateConfigUtil(filtered_normalized["eigenstate_config"])

    zero_point = [0, 0, 0]
    origin_point = [util.delta_x0[0] / 2, util.delta_x1[1] / 2, 0]
    next_origin_point = [-util.delta_x0[0] / 2, util.delta_x1[1] / 2, 0]
    points = [origin_point, next_origin_point, zero_point]
    for eigenstate in get_eigenstate_list(filtered):
        print(util.calculate_wavefunction_fast(eigenstate, points))
        print(np.abs(util.calculate_wavefunction_fast(eigenstate, points)))
    print("")
    for eigenstate in get_eigenstate_list(filtered_normalized):
        print(util.calculate_wavefunction_fast(eigenstate, points))
        print(np.abs(util.calculate_wavefunction_fast(eigenstate, points)))

    print(filtered_normalized["eigenstate_config"]["resolution"])

    eigenstates = get_eigenstate_list(filtered_normalized)
    fig, _, _ = plot_eigenstate_difference_in_z(
        filtered_normalized["eigenstate_config"], eigenstates[0], eigenstates[3]
    )
    fig.tight_layout()
    fig.show()

    path = get_data_path("eigenstates_grid_0.json")
    eigenstates = load_energy_eigenstates(path)
    normalized = normalize_eigenstate_phase_copper(eigenstates)
    filtered_normalized = filter_eigenstates_n_point(normalized, n=1)

    print(filtered_normalized["eigenstate_config"]["resolution"])

    eigenstates = get_eigenstate_list(filtered_normalized)
    fig, _, _ = plot_eigenstate_difference_in_z(
        filtered_normalized["eigenstate_config"], eigenstates[0], eigenstates[3]
    )
    fig.tight_layout()
    fig.show()

    fig, _, _ = plot_eigenstate_difference_in_z(
        filtered_normalized["eigenstate_config"], eigenstates[1], eigenstates[2]
    )
    fig.tight_layout()
    fig.show()

    input()






