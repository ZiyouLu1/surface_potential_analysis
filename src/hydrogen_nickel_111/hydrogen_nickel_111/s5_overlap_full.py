from __future__ import annotations

from surface_potential_analysis.overlap.calculation import calculate_wavepacket_overlap
from surface_potential_analysis.overlap.overlap import save_overlap

from .s4_wavepacket import (
    load_two_point_normalized_nickel_wavepacket_momentum as load_wavepacket,
)
from .surface_data import get_data_path


def calculate_overlap_nickel() -> None:
    for i in range(6):
        for j in range(i, 6):
            for dx0, dx1 in [(-1, -1), (-1, 0), (-1, 1), (0, 0), (0, 1), (1, 1)]:
                wavepacket_i = load_wavepacket(i)
                wavepacket_j = load_wavepacket(j, (dx0, dx1))
                overlap_ij = calculate_wavepacket_overlap(wavepacket_i, wavepacket_j)
                path = get_data_path(f"overlap/overlap_{i}_{j}_{dx0 % 3}_{dx1 % 3}.npy")
                save_overlap(path, overlap_ij)
