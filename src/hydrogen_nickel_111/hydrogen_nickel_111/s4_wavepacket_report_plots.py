from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.state_vector.conversion import (
    convert_state_vector_to_position_basis,
)
from surface_potential_analysis.wavepacket.get_eigenstate import get_eigenstate
from surface_potential_analysis.wavepacket.localization import (
    localize_tightly_bound_wavepacket_idx,
)
from surface_potential_analysis.wavepacket.plot import plot_wavepacket_2d_x

from hydrogen_nickel_111.s4_wavepacket import get_wavepacket


def plot_wavepacket_localization() -> None:
    fig, ax = plt.subplots()

    wavepacket = get_wavepacket(0)
    converted = convert_state_vector_to_position_basis(get_eigenstate(wavepacket, 0))
    idx_flat = np.argmax(np.abs(converted["vector"]), axis=-1)
    util = BasisUtil(converted["basis"])
    idx_max = util.get_stacked_index(idx_flat)

    normalized = localize_tightly_bound_wavepacket_idx(wavepacket, idx_max)
    fig, ax, mesh = plot_wavepacket_2d_x(
        normalized, (0, 1), (idx_max[2],), scale="symlog"
    )
    fig.show()

    wavepacket = get_wavepacket(0)
    print(wavepacket["vectors"].shape)
    normalized = localize_tightly_bound_wavepacket_idx(wavepacket, (8, 8, idx_max[2]))
    fig, ax, mesh = plot_wavepacket_2d_x(
        normalized, (0, 1), (idx_max[2],), scale="symlog"
    )
    fig.show()

    input()
