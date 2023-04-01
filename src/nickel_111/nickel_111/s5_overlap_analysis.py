import numpy as np
import scipy.optimize
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from surface_potential_analysis.overlap_transform import (
    OverlapTransform,
    load_overlap_transform,
)
from surface_potential_analysis.overlap_transform_plot import (
    plot_overlap_transform_along_diagonal,
    plot_overlap_transform_x0z,
    plot_overlap_transform_xy,
    plot_overlap_xy,
)
from surface_potential_analysis.surface_config import get_surface_xy_points

from .surface_data import get_data_path, save_figure


def get_max_point(overlap: OverlapTransform) -> tuple[int, int, int]:
    points = np.asarray(overlap["points"])
    (ikx0, ikx1, inz) = np.unravel_index(np.argmax(np.abs(points)), shape=points.shape)
    return (int(ikx0), int(ikx1), int(inz))


def make_transform_real_at(
    overlap: OverlapTransform, point: tuple[int, int, int] | None = None
) -> OverlapTransform:
    """
    Shift the phase of the overlap transform such tha tis it real at point
    This is equivalent to shifting the origin of the real space grid we use when
    calculating the overlap.

    Parameters
    ----------
    overlap : OverlapTransform
    point : tuple[int, int, int] | None, optional
        Point to make real, by default the maximum of the transform

    Returns
    -------
    OverlapTransform
        A new overlap transform, which is real at the given point
    """
    point = get_max_point(overlap) if point is None else point

    new_points = overlap["points"] * np.exp(-1j * np.angle(overlap["points"][point]))
    return {
        "dkx0": overlap["dkx0"],
        "dkx1": overlap["dkx1"],
        "dkz": overlap["dkz"],
        "points": new_points,
    }


def plot_overlap():
    path = get_data_path("overlap_transform_orthogonal_hcp_fcc.npz")
    # path = get_data_path("overlap_transform_interpolated_hcp_fcc.npz")
    # path = get_data_path("overlap_transform_extended_hcp_fcc.npz")
    overlap = load_overlap_transform(path)
    # overlap = make_transform_real_at(overlap, point=(1, 1, 0))
    fig, ax, _ = plot_overlap_transform_xy(overlap)
    ax.set_title(
        "Plot of the overlap transform for ikz=0\n"
        "showing oscillation in the direction corresponding to\n"
        "a vector spanning the fcc and hcp sites"
    )
    save_figure(fig, "2d_overlap_transform_kx_ky.png")
    fig.show()

    fig, ax, _ = plot_overlap_transform_xy(overlap, measure="real")
    ax.set_title(
        "Plot of the overlap transform for ikz=0\n"
        "showing oscillation in the direction corresponding to\n"
        "a vector spanning the fcc and hcp sites"
    )
    save_figure(fig, "2d_overlap_transform_real_kx_ky.png")
    fig.show()

    fig, ax, _ = plot_overlap_transform_xy(overlap, measure="imag")
    ax.set_title(
        "Plot of the overlap transform for ikz=0\n"
        "showing oscillation in the direction corresponding to\n"
        "a vector spanning the fcc and hcp sites"
    )
    save_figure(fig, "2d_overlap_transform_imag_kx_ky.png")
    fig.show()

    fig, ax, _ = plot_overlap_xy(overlap)
    ax.set_title(
        "Plot of the overlap summed over z\n"
        "showing the FCC and HCP asymmetry\n"
        "in a small region in the center of the figure"
    )
    save_figure(fig, "2d_overlap_kx_ky.png")
    fig.show()

    fig, ax, _ = plot_overlap_xy(overlap, measure="real")
    ax.set_title(
        "Plot of the overlap summed over z\n"
        "showing the FCC and HCP asymmetry\n"
        "in a small region in the center of the figure"
    )
    save_figure(fig, "2d_overlap_real_kx_ky.png")
    fig.show()

    fig, ax, _ = plot_overlap_transform_x0z(overlap)
    ax.set_title(
        "Plot of the overlap transform for ikx1=0\n"
        "A very sharp peak in the kz direction"
    )
    save_figure(fig, "2d_overlap_fraction_kx1_kz.png")
    fig.show()

    fig, ax = plt.subplots()
    _, _, ln = plot_overlap_transform_along_diagonal(overlap, measure="abs", ax=ax)
    ln.set_label("abs")
    _, _, ln = plot_overlap_transform_along_diagonal(overlap, measure="real", ax=ax)
    ln.set_label("real")
    _, _, ln = plot_overlap_transform_along_diagonal(overlap, measure="imag", ax=ax)
    ln.set_label("imag")

    delta_k = np.linalg.norm(
        np.add(
            np.multiply(overlap["dkx0"], overlap["points"].shape[0]),
            np.multiply(overlap["dkx1"], overlap["points"].shape[1]),
        )
    )
    k_points = np.linspace(0, delta_k, 5000)
    # fit_points = (
    #     0.75
    #     * np.max(np.abs(overlap["points"]))
    #     * np.exp(-((k_points - (delta_k / 2)) ** 2) / (delta_k / 8) ** 2)
    # )
    # ax.plot(k_points, fit_points)
    # fit_points = (
    #     1.1
    #     * np.max(np.abs(overlap["points"]))
    #     * np.exp(-((k_points - (delta_k / 2)) ** 2) / (delta_k / 7) ** 2)
    # )

    # ax.plot(k_points, fit_points)

    # fit_points = (
    #     0.4
    #     * np.max(np.abs(overlap["points"]))
    #     * np.exp(-((k_points - (delta_k / 2)) ** 2) / (delta_k / 7) ** 2)
    # )

    # ax.plot(k_points, fit_points)

    # fit_points = (
    #     (0.75 - 0.35 * np.cos(4 * np.pi * (k_points - (delta_k / 2)) / (delta_k / 8)))
    #     * np.max(np.abs(overlap["points"]))
    #     * np.exp(-((k_points - (delta_k / 2)) ** 2) / (delta_k / 7) ** 2)
    # )
    # ax.plot(k_points, fit_points)

    # fit_points = (
    #     (0.72 - 0.35 * np.cos(4 * np.pi * (k_points - (delta_k / 2)) / (delta_k / 8)))
    #     * np.min(np.real(overlap["points"]))
    #     * np.exp(-((k_points - (delta_k / 2)) ** 2) / (delta_k / 7) ** 2)
    # )
    # ax.plot(k_points, fit_points)

    # fit_points = (
    #     (
    #         -np.sign(k_points - (delta_k / 2)) * 0.27
    #         - 0.4 * np.sin(4 * np.pi * (k_points - (delta_k / 2)) / (delta_k / 8))
    #     )
    #     * np.min(np.real(overlap["points"]))
    #     * np.exp(-((k_points - (delta_k / 2)) ** 2) / (delta_k / 7) ** 2)
    # )
    # ax.plot(k_points, fit_points)

    # fit_points = (
    #     -2
    #     * ((k_points - (delta_k / 2)) / (delta_k / 8))
    #     * (0.72 - 0.35 * np.cos(4 * np.pi * (k_points - (delta_k / 2)) / (delta_k / 8)))
    #     * np.min(np.real(overlap["points"]))
    #     * np.exp(-((k_points - (delta_k / 2)) ** 2) / (delta_k / 7) ** 2)
    # )

    ax2 = ax.twinx()
    _, _, ln = plot_overlap_transform_along_diagonal(overlap, measure="angle", ax=ax2)
    ln.set_label("angle")
    fit_points = (
        -2
        * ((k_points - (delta_k / 2)) / (delta_k / 8))
        * (0.72 - 0.35 * np.cos(4 * np.pi * (k_points - (delta_k / 2)) / (delta_k / 8)))
        * np.min(np.real(overlap["points"]))
        * np.exp(-((k_points - (delta_k / 2)) ** 2) / (delta_k / 7) ** 2)
    )

    def fit_curve(
        kxy: tuple[NDArray, NDArray], dkx: float, dky: float, A0: float, alpha: float
    ):
        (kx, ky) = kxy
        return (
            (np.sin(kx / dkx + ky / dky) ** 2)
            * A0
            * np.exp(-(kx**2 + ky**2) / alpha**2)
        )

    ax2.plot(k_points, fit_points)

    ax.legend()
    ax.set_title(
        "Plot of the wavefunction along the diagonal,\nshowing an oscillation in the overlap"
    )
    save_figure(fig, "diagonal_1d_overlap_fraction.png")
    fig.show()
    input()


def fit_overlap_transform():
    path = get_data_path("overlap_transform_hcp_fcc.npz")
    overlap = load_overlap_transform(path)
    points = overlap["points"]

    print(points[0, 0, 0])
    print(np.max(np.abs(points[:, :, 0])))
    print(np.max(np.abs(points[:, :])))

    print(points.shape)
    print((points[177, 177, 0]))
    print((points[177, 178, 0]))
    print((points[178, 177, 0]))
    print((points[178, 178, 0]))

    print(get_max_point(overlap))
    (ikx0, ikx1, _) = get_max_point(overlap)
    ikx0 = 184 - ikx0
    ikx1 = 184 - ikx1
    kmax = (
        (overlap["dkx0"][0] * ikx0) + (overlap["dkx1"][0] * ikx1),
        (overlap["dkx0"][1] * ikx0) + (overlap["dkx1"][1] * ikx1),
    )
    print(kmax, np.linalg.norm(kmax))

    def fit_curve(
        kxy: tuple[NDArray, NDArray], dkx: float, dky: float, A0: float, alpha: float
    ):
        (kx, ky) = kxy
        return (
            (np.sin(kx / dkx + ky / dky) ** 2)
            * A0
            * np.exp(-(kx**2 + ky**2) / alpha**2)
        )

    offset = (
        -overlap["dkx0"][0] * (points.shape[0] // 2)
        - overlap["dkx1"][0] * (points.shape[1] // 2),
        -overlap["dkx0"][1] * (points.shape[0] // 2)
        - overlap["dkx1"][1] * (points.shape[1] // 2),
    )

    print(offset)

    coordinates = get_surface_xy_points(
        {
            "delta_x0": (
                overlap["dkx0"][0] * points.shape[0],
                overlap["dkx0"][1] * points.shape[0],
            ),
            "delta_x1": (
                overlap["dkx1"][0] * points.shape[1],
                overlap["dkx1"][1] * points.shape[1],
            ),
        },
        shape=(points.shape[0], points.shape[1]),
    )
    # TODO: maybe ifftshift
    amplitudes = np.fft.fftshift(np.abs(points[:, :, 0]))

    (ikx0, ikx1) = np.unravel_index(np.argmax(amplitudes), shape=amplitudes.shape)
    print(ikx0, ikx1)
    print(coordinates[92, 92])
    print(coordinates[ikx0, ikx1])
    coordinates -= coordinates[92, 92]
    print(coordinates[92, 92])
    print(coordinates[ikx0, ikx1])

    kx_coord = coordinates[:, :, 0].ravel()
    ky_coord = coordinates[:, :, 1].ravel()
    popt, pcov = scipy.optimize.curve_fit(
        fit_curve,
        (kx_coord, ky_coord),
        amplitudes.ravel(),
        p0=[
            coordinates[ikx0, ikx1][0],
            coordinates[ikx0, ikx1][1],
            np.max(amplitudes),
            np.linalg.norm(overlap["dkx1"]),
        ],
    )
    print("popt", popt)
    print("pcov", pcov)

    input()


def calculate_max_overlap_transform(overlap: OverlapTransform):
    points = overlap["points"]
    xy_points = get_surface_xy_points(
        {
            "delta_x0": (
                overlap["dkx0"][0] * points.shape[0],
                overlap["dkx0"][1] * points.shape[0],
            ),
            "delta_x1": (
                overlap["dkx1"][0] * points.shape[1],
                overlap["dkx1"][1] * points.shape[1],
            ),
        },
        shape=(points.shape[0], points.shape[1]),
    )
    (ikx0, ikx1, ikz) = np.unravel_index(np.argmax(np.abs(points)), shape=points.shape)
    xy_points -= xy_points[points.shape[0] // 2, points.shape[1] // 2]

    xy_point = xy_points[ikx0, ikx1]
    return points[ikx0, ikx1, ikz], xy_point


def print_max_overlaps():
    path = get_data_path("overlap_transform_orthogonal_hcp_fcc.npz")
    overlap = load_overlap_transform(path)
    max, idx = calculate_max_overlap_transform(overlap)
    print(max, np.abs(max), idx)

    path = get_data_path("overlap_transform_orthogonal_fcc_fcc.npz")
    overlap = load_overlap_transform(path)
    max, idx = calculate_max_overlap_transform(overlap)
    print(max, np.abs(max), idx)

    path = get_data_path("overlap_transform_orthogonal_hcp_hcp.npz")
    overlap = load_overlap_transform(path)
    max, idx = calculate_max_overlap_transform(overlap)
    print(max, np.abs(max), idx)
