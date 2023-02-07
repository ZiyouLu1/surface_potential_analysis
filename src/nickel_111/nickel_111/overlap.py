import numpy as np

from surface_potential_analysis.wavepacket_grid import (
    calculate_volume_element,
    load_wavepacket_grid_legacy,
)

from .surface_data import get_data_path


def calculate_overlap_factor():
    path = get_data_path("eigenstates_wavepacket_0.json")
    wavepacket0 = load_wavepacket_grid_legacy(path)

    path = get_data_path("eigenstates_wavepacket_1.json")
    wavepacket1 = load_wavepacket_grid_legacy(path)

    dv = calculate_volume_element(wavepacket0)

    points0 = np.array(wavepacket0["points"])
    print(dv * np.sum(np.square(np.abs(points0))))

    points1 = np.array(wavepacket1["points"])
    print(dv * np.sum(np.square(np.abs(points1))))

    print(dv * np.sum(np.abs(points0) * np.abs(points1)))
    print(dv * np.abs(np.sum(points0 * np.conj(points1))))
