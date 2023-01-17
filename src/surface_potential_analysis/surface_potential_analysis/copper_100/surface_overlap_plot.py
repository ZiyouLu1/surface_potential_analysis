from typing import List

import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import AxesImage
from numpy.typing import NDArray

from ..energy_data.plot_wavepacket_grid import plot_wavepacket_grid_xy
from ..energy_data.wavepacket_grid import (
    calculate_volume_element,
    load_wavepacket_grid,
    mask_negative_wavepacket,
    symmetrize_wavepacket,
)
from .surface_data import get_data_path, save_figure


def plot_overlap_factor():
    path = get_data_path("copper_eigenstates_wavepacket.json")
    wavepacket = symmetrize_wavepacket(load_wavepacket_grid(path))

    dv = calculate_volume_element(wavepacket)

    points = np.array(wavepacket["points"])
    points1 = points[0:65]
    points2 = points[32:97]
    print(dv * np.sum(np.square(np.abs(points))))
    print(dv * np.sum(np.abs(points1) * np.abs(points2)))
    print(points.shape)

    fig, _, img = plot_wavepacket_grid_xy(wavepacket, z_ind=10, measure="real")
    img.set_norm("symlog")  # type: ignore
    fig.show()

    path = get_data_path("copper_eigenstates_wavepacket_interpolated.json")
    # interpolated = interpolate_wavepacket(wavepacket, (193, 193, 101))
    # save_wavepacket_grid(interpolated, path)
    interpolated = load_wavepacket_grid(path)

    dv = calculate_volume_element(interpolated)
    points = np.array(interpolated["points"])
    points1 = points[0:129]
    points2 = points[64:193]
    print(dv * np.sum(np.square(np.abs(points))))
    print(dv * np.sum(np.abs(points1) * np.abs(points2)))
    print(points.shape)

    fig, _, img = plot_wavepacket_grid_xy(interpolated, z_ind=50, measure="real")
    img.set_norm("symlog")  # type: ignore
    fig.show()
    save_figure(fig, "Plot of interpolated wavepacket grid at z=0")


def plot_masked_overlap_factor():
    # path = get_data_path("copper_eigenstates_wavepacket_4_point_interpolated.json")
    # interpolated = load_wavepacket_grid(path)
    path = get_data_path("copper_eigenstates_wavepacket.json")
    wavepacket = symmetrize_wavepacket(load_wavepacket_grid(path))

    masked = mask_negative_wavepacket(wavepacket)
    dv = calculate_volume_element(masked)
    points = np.array(masked["points"])
    points1 = points[0:65]
    points2 = points[32:97]
    print(dv * np.sum(np.square(np.abs(points))))
    print(dv * np.sum(np.abs(points1[:, :, :]) * np.abs(points2[:, :, :])))

    overlap_fraction = np.abs(points1) * np.abs(points2)
    max_index = np.unravel_index(
        np.argmax(overlap_fraction.ravel()), overlap_fraction.shape
    )
    masked_overlap = np.copy(overlap_fraction)
    masked_overlap[masked_overlap < 1000] = 0
    print(dv * np.sum(masked_overlap))

    print(overlap_fraction.shape, max_index)

    fig, axs = plt.subplots(1, 2)
    _, _, img = plot_wavepacket_grid_xy(
        masked, z_ind=overlap_fraction.shape[2] // 2, measure="real", ax=axs[0]
    )
    axs[0].set_title("Z=0")
    img.set_norm("symlog")  # type: ignore

    _, _, img = plot_wavepacket_grid_xy(
        masked, z_ind=int(max_index[2]), measure="real", ax=axs[1]
    )
    img.set_norm("symlog")  # type: ignore
    axs[1].set_title("Max overlap z")
    fig.show()
    save_figure(fig, "Z=0 and max overlap height wavefunction comparison")

    fig, ax1 = plt.subplots()
    ax1.plot(masked["x_points"][0:65], overlap_fraction[:, max_index[1], max_index[2]])
    ax1.set_ylabel("Overlap Fraction")
    ax1.set_title(
        f"Interpolated overlap in x, at maximum z (z = {masked['z_points'][max_index[2]]})\n"
        f"and y symmetry point"
    )

    ax2 = ax1.twinx()
    ax2.plot(masked["x_points"][0:65], points[0:65, max_index[1], max_index[2]])
    ax2.plot(masked["x_points"][0:65], points[32:97, max_index[1], max_index[2]])
    ax2.set_ylabel("Wavefunction")
    ax2.set_yscale("symlog")
    fig.show()
    save_figure(fig, "overlap factor at symmetry point")

    for z in range(11, 15):
        fig, ax1 = plt.subplots()
        ax1.plot(masked["x_points"][0:65], overlap_fraction[:, max_index[1], z])
        ax1.set_ylabel("Overlap Fraction")
        ax1.set_title(
            f"Interpolated overlap in x, at maximum z (z = {masked['z_points'][z]})\n"
            f"and y symmetry point"
        )

        ax2 = ax1.twinx()
        ax2.plot(masked["x_points"][0:65], points[0:65, max_index[1], z])
        ax2.plot(masked["x_points"][0:65], points[32:97, max_index[1], z])
        ax2.set_ylabel("Wavefunction")
        ax2.set_yscale("symlog")
        fig.show()


def plot_overlap_fraction_2D(overlap_fraction: NDArray):

    max_index = np.unravel_index(
        np.argmax(overlap_fraction.ravel()), overlap_fraction.shape
    )
    max_fraction = np.max(overlap_fraction)
    print(overlap_fraction.shape, max_index)

    fig, ax = plt.subplots(2, 3)

    im = ax[0][0].imshow(overlap_fraction[:, :, 9])
    im.set_norm("symlog")  # type: ignore
    im.set_clim(0, max_fraction)

    im = ax[0][1].imshow(overlap_fraction[:, :, 10])
    im.set_norm("symlog")  # type: ignore
    im.set_clim(0, max_fraction)

    im = ax[0][2].imshow(overlap_fraction[:, :, 11])
    im.set_norm("symlog")  # type: ignore
    im.set_clim(0, max_fraction)

    im = ax[1][0].imshow(overlap_fraction[:, :, 12])
    im.set_norm("symlog")  # type: ignore
    im.set_clim(0, max_fraction)

    im = ax[1][1].imshow(overlap_fraction[:, :, 13])
    im.set_norm("symlog")  # type: ignore
    im.set_clim(0, max_fraction)

    im = ax[1][2].imshow(overlap_fraction[:, :, 14])
    im.set_norm("symlog")  # type: ignore
    im.set_clim(0, max_fraction)

    fig.suptitle("Plot of overlap fraction in the xy plane")
    fig.show()
    save_figure(fig, "overlap fraction in xy plane")

    fig, axs = plt.subplots()
    ims: List[List[AxesImage]] = []
    for z in range(overlap_fraction.shape[2]):
        im = axs.imshow(overlap_fraction[:, :, z], animated=True)
        im.set_norm("symlog")  # type: ignore
        im.set_clim(0, max_fraction)
        ims.append([im])
    im = axs.imshow(overlap_fraction[:, :, 0])
    im.set_norm("symlog")  # type: ignore
    im.set_clim(0, max_fraction)
    ani = matplotlib.animation.ArtistAnimation(
        fig, ims, interval=100, blit=True, repeat_delay=50
    )
    fig.show()

    input()


def plot_8_point_overlap_fraction():
    path = get_data_path("copper_eigenstates_wavepacket.json")
    wavepacket = symmetrize_wavepacket(load_wavepacket_grid(path))

    points = np.array(wavepacket["points"])
    points1 = points[0:65]
    points2 = points[32:97]
    overlap_fraction = np.abs(points1) * np.abs(points2)
    plot_overlap_fraction_2D(overlap_fraction)


def plot_overlap_fraction_corrected():
    """
    Attempt to 'correct' the overlap fraction by removing the overlap
    past the next neighboring site.
    Since the extra overlap is symmetrical about this point we also
    remove the overlap both sides of the
    """

    path = get_data_path("copper_eigenstates_wavepacket.json")
    wavepacket = symmetrize_wavepacket(load_wavepacket_grid(path))

    points = np.array(wavepacket["points"])
    points1 = points[0:65]
    points2 = points[32:97]
    overlap_fraction = np.abs(points1) * np.abs(points2)

    print(overlap_fraction.shape)

    overlap_fraction[16, :, :] = 0
    overlap_fraction[17:33] = (
        overlap_fraction[17:33] - overlap_fraction[0:16:, :, :][::-1]
    )
    overlap_fraction[0:16, :, :] = 0

    overlap_fraction[48, :, :] = 0
    overlap_fraction[33:48] = (
        overlap_fraction[33:48] - overlap_fraction[49:-1, :, :][::-1]
    )
    overlap_fraction[49:, :, :] = 0

    overlap_fraction[overlap_fraction < 0] = 0

    dv = calculate_volume_element(wavepacket)
    print(dv * np.sum(np.square(np.abs(points))))
    print(dv * np.sum(overlap_fraction))

    plot_overlap_fraction_2D(overlap_fraction)
