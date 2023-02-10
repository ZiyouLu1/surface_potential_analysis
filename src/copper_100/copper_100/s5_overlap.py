import numpy as np

from surface_potential_analysis.wavepacket_grid import (
    WavepacketGrid,
    calculate_volume_element,
    load_wavepacket_grid,
    load_wavepacket_grid_legacy,
    reflect_wavepacket_in_axis,
    symmetrize_wavepacket_about_far_edge,
)

from .surface_data import get_data_path


def calculate_overlap_factor():
    path = get_data_path("copper_eigenstates_wavepacket.json")
    wavepacket = symmetrize_wavepacket_about_far_edge(load_wavepacket_grid_legacy(path))

    dv = calculate_volume_element(wavepacket)

    points = np.array(wavepacket["points"])
    points1 = points[0:65]
    points2 = points[32:97]
    print(dv * np.sum(np.square(np.abs(points))))
    print(dv * np.sum(np.abs(points1) * np.abs(points2)))
    print(points.shape)

    path = get_data_path("copper_eigenstates_wavepacket_interpolated.json")
    # interpolated = interpolate_wavepacket(wavepacket, (193, 193, 101))
    # save_wavepacket_grid(interpolated, path)
    interpolated = load_wavepacket_grid_legacy(path)

    dv = calculate_volume_element(interpolated)
    points = np.array(interpolated["points"])
    points1 = points[0:129]
    points2 = points[64:193]
    print(dv * np.sum(np.square(np.abs(points))))
    print(dv * np.sum(np.abs(points1) * np.abs(points2)))
    print(points.shape)


def mask_interpolation_artifacts_copper_100(grid: WavepacketGrid) -> WavepacketGrid:
    above_threshold = np.abs(grid["points"]) > 1e-6 * np.abs(np.max(grid["points"]))
    keep_points_lim = np.argmin(above_threshold, axis=0)
    x_indices = np.indices(dimensions=above_threshold.shape)[0]
    return {
        "points": np.where(x_indices < keep_points_lim, grid["points"], 0).tolist(),
        "delta_x0": grid["delta_x0"],
        "delta_x1": grid["delta_x1"],
        "delta_z": grid["delta_z"],
    }


def calculate_overlap_factor_relaxed():
    path = get_data_path("relaxed_eigenstates_wavepacket.json")
    wavepacket = reflect_wavepacket_in_axis(load_wavepacket_grid(path), axis=2)
    dv = calculate_volume_element(wavepacket)
    print(dv)

    points = np.array(wavepacket["points"])
    points1 = points
    points2 = points[::-1]
    print(dv * np.sum(np.square(np.abs(points))))
    print(dv * np.sum(np.abs(points1) * np.abs(points2)))
    print(points.shape)

    masked = mask_interpolation_artifacts_copper_100(wavepacket)
    points = np.array(masked["points"])
    points1 = points
    points2 = points[::-1]
    print(dv * np.sum(np.square(np.abs(points))))
    print(dv * np.sum(np.abs(points1) * np.abs(points2)))
    print(np.sum(np.abs(points1) * np.abs(points2)))
    print(points.shape)
