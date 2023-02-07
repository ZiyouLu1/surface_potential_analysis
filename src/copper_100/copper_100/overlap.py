import numpy as np

from surface_potential_analysis.wavepacket_grid import (
    calculate_volume_element,
    load_wavepacket_grid_legacy,
    symmetrize_wavepacket,
)

from .surface_data import get_data_path


def calculate_overlap_factor():
    path = get_data_path("copper_eigenstates_wavepacket.json")
    wavepacket = symmetrize_wavepacket(load_wavepacket_grid_legacy(path))

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
