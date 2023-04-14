from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from surface_potential_analysis.wavepacket import save_wavepacket
from surface_potential_analysis.wavepacket.wavepacket import generate_wavepacket

from .s2_hamiltonian import generate_hamiltonian_sho
from .surface_data import get_data_path

if TYPE_CHECKING:
    from surface_potential_analysis.hamiltonian import Hamiltonian


def generate_wavepacket_sho() -> None:
    def hamiltonian_generator(
        x: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    ) -> Hamiltonian[Any]:
        return generate_hamiltonian_sho(
            shape=(46, 46, 100),
            bloch_phase=x,
            resolution=(23, 23, 16),
        )

    save_bands = np.arange(20)

    wavepackets = generate_wavepacket(
        hamiltonian_generator, samples=(8, 8), save_bands=save_bands
    )
    for k, wavepacket in zip(save_bands, wavepackets, strict=True):
        path = get_data_path(f"wavepacket_{k}.npy")
        save_wavepacket(path, wavepacket)
