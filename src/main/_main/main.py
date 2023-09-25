from __future__ import annotations

import numpy as np
from hydrogen_copper_100.s4_wavepacket import get_localized_hamiltonian_hydrogen
from hydrogen_copper_100.s4_wavepacket_plot import plot_localized_wavepackets_hydrogen
from surface_potential_analysis.util.decorators import timed


@timed
def main() -> None:
    a = get_localized_hamiltonian_hydrogen()
    b = a["data"].reshape(-1, *a["basis"][1].shape)
    _c = np.diagonal(b, axis1=1, axis2=2)
    plot_localized_wavepackets_hydrogen()


if __name__ == "__main__":
    main()
