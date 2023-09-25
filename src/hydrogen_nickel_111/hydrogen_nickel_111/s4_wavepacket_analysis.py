from __future__ import annotations

import numpy as np
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.state_vector.conversion import (
    convert_state_vector_to_position_basis,
)
from surface_potential_analysis.wavepacket.get_eigenstate import (
    get_wavepacket_state_vector,
)

from .s4_wavepacket import get_wavepacket_hydrogen


def calculate_wavepacket_maximums() -> None:
    """Calculate the maximum of the k=0 eigenstate of a wavepacket for each band."""
    for band in range(20):
        wavepacket = get_wavepacket_hydrogen(band)

        state = get_wavepacket_state_vector(wavepacket, 0)
        converted = convert_state_vector_to_position_basis(state)  # type: ignore[arg-type] # ive done variance wrong somewhere
        util = BasisUtil(converted["basis"])

        print(f"Band {band}")  # noqa: T201
        print(util.get_stacked_index(int(np.argmax(converted["data"]))))  # noqa: T201
