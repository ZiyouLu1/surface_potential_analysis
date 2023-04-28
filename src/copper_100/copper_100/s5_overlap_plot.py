from __future__ import annotations

import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np

from .surface_data import save_figure


def symmetrize_wavepacket_about_far_edge(wavepacket: WavepacketGrid) -> WavepacketGrid:
    points = np.array(wavepacket["points"])

    reflected_shape = (
        points.shape[0] * 2 - 1,
        points.shape[1] * 2 - 1,
        points.shape[2],
    )
    reflected_points = np.zeros(reflected_shape, dtype=complex)
    reflected_points[: points.shape[0], : points.shape[1]] = points[:, :]
    reflected_points[points.shape[0] - 1 :, : points.shape[1]] = points[::-1, :]
    reflected_points[: points.shape[0], points.shape[1] - 1 :] = points[:, ::-1]
    reflected_points[points.shape[0] - 1 :, points.shape[1] - 1 :] = points[::-1, ::-1]

    return {
        "points": reflected_points.tolist(),
        "delta_x0": (wavepacket["delta_x0"][0] * 2, wavepacket["delta_x0"][1] * 2),
        "delta_x1": (wavepacket["delta_x1"][0] * 2, wavepacket["delta_x1"][1] * 2),
        "z_points": wavepacket["z_points"],
    }


def plot_overlap_fraction_2d(overlap_fraction: NDArray):
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
    ims: list[list[AxesImage]] = []
    for z in range(overlap_fraction.shape[2]):
        im = axs.imshow(overlap_fraction[:, :, z], animated=True)
        im.set_norm("symlog")  # type: ignore
        im.set_clim(0, max_fraction)
        ims.append([im])
    im = axs.imshow(overlap_fraction[:, :, 0])
    im.set_norm("symlog")  # type: ignore
    im.set_clim(0, max_fraction)
    _ani = matplotlib.animation.ArtistAnimation(
        fig, ims, interval=100, blit=True, repeat_delay=50
    )
    fig.show()

    input()
