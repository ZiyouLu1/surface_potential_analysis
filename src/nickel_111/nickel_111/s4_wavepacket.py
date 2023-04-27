from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from surface_potential_analysis.wavepacket import save_wavepacket
from surface_potential_analysis.wavepacket.wavepacket import (
    Wavepacket,
    generate_wavepacket,
    load_wavepacket,
)

from .s2_hamiltonian import generate_hamiltonian_sho
from .surface_data import get_data_path

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import (
        ExplicitBasis,
        MomentumBasis,
        PositionBasis,
        TruncatedBasis,
    )
    from surface_potential_analysis.basis_config.basis_config import BasisConfig
    from surface_potential_analysis.hamiltonian import Hamiltonian


def load_nickel_wavepacket(
    idx: int,
) -> Wavepacket[
    Literal[8],
    Literal[8],
    BasisConfig[
        TruncatedBasis[Literal[23], MomentumBasis[Literal[250]]],
        TruncatedBasis[Literal[23], MomentumBasis[Literal[250]]],
        ExplicitBasis[Literal[12], PositionBasis[Literal[250]]],
    ],
]:
    path = get_data_path(f"wavepacket_large_{idx}.npy")
    return load_wavepacket(path)


def generate_nickel_wavepacket_sho() -> None:
    def hamiltonian_generator(
        x: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    ) -> Hamiltonian[Any]:
        return generate_hamiltonian_sho(
            shape=(250, 250, 250),
            bloch_phase=x,
            resolution=(23, 23, 12),
        )

    save_bands = np.arange(20)

    wavepackets = generate_wavepacket(
        hamiltonian_generator,
        samples=(12, 12),
        save_bands=save_bands,
    )
    for k, wavepacket in zip(save_bands, wavepackets, strict=True):
        path = get_data_path(f"wavepacket_large_{k}.npy")
        save_wavepacket(path, wavepacket)
