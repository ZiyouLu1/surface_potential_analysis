from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from surface_potential_analysis.basis_config.basis_config import (
    BasisConfig,
    BasisConfigUtil,
    MomentumBasisConfig,
)
from surface_potential_analysis.wavepacket import save_wavepacket
from surface_potential_analysis.wavepacket.conversion import convert_wavepacket_to_basis
from surface_potential_analysis.wavepacket.normalization import normalize_wavepacket
from surface_potential_analysis.wavepacket.wavepacket import (
    Wavepacket,
    generate_wavepacket,
    load_wavepacket,
)

from .s2_hamiltonian import generate_hamiltonian_sho
from .surface_data import get_data_path

if TYPE_CHECKING:
    from surface_potential_analysis._types import SingleIndexLike
    from surface_potential_analysis.basis.basis import (
        ExplicitBasis,
        MomentumBasis,
        PositionBasis,
        TruncatedBasis,
    )
    from surface_potential_analysis.hamiltonian import Hamiltonian


MAXIMUM_POINTS: list[tuple[int, int, int]] = [
    (0, 0, 102),
    (8, 8, 103),
    (1, 2, 108),
    (21, 2, 108),
    (11, 7, 109),
    (11, 6, 109),
    (20, 2, 118),
    (0, 0, 85),
    (8, 8, 86),
    (17, 4, 116),
    (16, 4, 116),
    (0, 0, 114),
    (23, 22, 112),
    (23, 3, 112),
    (1, 1, 95),
    (22, 1, 95),
    (7, 7, 96),
    (7, 11, 98),
    (6, 6, 104),
    (1, 2, 123),
]


def load_copper_wavepacket(
    band: int,
) -> Wavepacket[
    Literal[12],
    Literal[12],
    BasisConfig[
        TruncatedBasis[Literal[24], MomentumBasis[Literal[24]]],
        TruncatedBasis[Literal[24], MomentumBasis[Literal[24]]],
        ExplicitBasis[Literal[12], PositionBasis[Literal[250]]],
    ],
]:
    path = get_data_path(f"wavepacket_{band}.npy")
    wavepacket = load_wavepacket(path)
    wavepacket["basis"][0]["parent"]["n"] = 24
    wavepacket["basis"][1]["parent"]["n"] = 24
    return wavepacket


def load_normalized_copper_wavepacket_momentum(
    band: int, idx: SingleIndexLike = 0, angle: float = 0
) -> Wavepacket[
    Literal[12],
    Literal[12],
    MomentumBasisConfig[Literal[24], Literal[24], Literal[250]],
]:
    wavepacket = load_copper_wavepacket(band)
    util = BasisConfigUtil(wavepacket["basis"])
    basis: MomentumBasisConfig[Literal[24], Literal[24], Literal[250]] = (
        {"_type": "momentum", "delta_x": util.delta_x0, "n": 24},
        {"_type": "momentum", "delta_x": util.delta_x1, "n": 24},
        {"_type": "momentum", "delta_x": util.delta_x2, "n": 250},
    )
    normalized = normalize_wavepacket(wavepacket, idx, angle)
    return convert_wavepacket_to_basis(normalized, basis)


def generate_wavepacket_sho() -> None:
    def hamiltonian_generator(
        x: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    ) -> Hamiltonian[Any]:
        return generate_hamiltonian_sho(
            shape=(48, 48, 250),
            bloch_phase=x,
            resolution=(24, 24, 16),
        )

    save_bands = np.arange(20)

    wavepackets = generate_wavepacket(
        hamiltonian_generator, samples=(12, 12), save_bands=save_bands
    )
    for k, wavepacket in zip(save_bands, wavepackets, strict=True):
        path = get_data_path(f"wavepacket_{k}.npy")
        save_wavepacket(path, wavepacket)
