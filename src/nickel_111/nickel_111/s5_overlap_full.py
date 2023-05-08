from __future__ import annotations

from surface_potential_analysis.overlap.calculation import calculate_wavepacket_overlap
from surface_potential_analysis.overlap.overlap import save_overlap
from surface_potential_analysis.wavepacket.normalization import calculate_normalisation

from .s4_wavepacket import (
    load_two_point_normalized_nickel_wavepacket_momentum as load_wavepacket,
)
from .surface_data import get_data_path


def calculate_overlap_nickel() -> None:
    wavepackets = [load_wavepacket(band) for band in range(6)]

    print([calculate_normalisation(w) for w in wavepackets])  # noqa: T201

    for i, wavepacket_i in enumerate(wavepackets):
        for j, wavepacket_j in enumerate(wavepackets[i + 1 :]):
            overlap_ij = calculate_wavepacket_overlap(wavepacket_i, wavepacket_j)
            path = get_data_path(f"overlap_{i}_{j}.npy")
            save_overlap(path, overlap_ij)
