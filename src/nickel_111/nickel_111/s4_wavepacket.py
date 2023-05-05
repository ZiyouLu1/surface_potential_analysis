from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from surface_potential_analysis.basis_config.basis_config import BasisConfigUtil
from surface_potential_analysis.wavepacket import save_wavepacket
from surface_potential_analysis.wavepacket.conversion import convert_wavepacket_to_basis
from surface_potential_analysis.wavepacket.normalization import (
    normalize_wavepacket,
    normalize_wavepacket_two_point,
)
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
    from surface_potential_analysis.basis_config.basis_config import (
        BasisConfig,
        MomentumBasisConfig,
    )
    from surface_potential_analysis.hamiltonian import Hamiltonian

MAXIMUM_POINTS: list[tuple[int, int, int]] = [
    (0, 0, 117),
    (8, 8, 117),
    (2, 21, 124),
    (1, 1, 117),
    (6, 11, 124),
    (7, 7, 117),
    (4, 4, 130),
    (0, 0, 100),
    (8, 7, 101),
    (16, 4, 130),
    (4, 4, 130),
    (8, 8, 133),
    (23, 3, 125),
    (5, 5, 128),
    (1, 1, 108),
    (21, 1, 115),
    (7, 7, 107),
    (10, 7, 110),
    (8, 8, 126),
    (20, 1, 140),
]


def load_nickel_wavepacket(
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


def load_normalized_nickel_wavepacket_momentum(
    band: int, idx: int | tuple[int, int, int] = 0, angle: float = 0
) -> Wavepacket[
    Literal[12],
    Literal[12],
    MomentumBasisConfig[Literal[24], Literal[24], Literal[250]],
]:
    wavepacket = load_nickel_wavepacket(band)
    util = BasisConfigUtil(wavepacket["basis"])
    basis: MomentumBasisConfig[Literal[24], Literal[24], Literal[250]] = (
        {"_type": "momentum", "delta_x": util.delta_x0, "n": 24},
        {"_type": "momentum", "delta_x": util.delta_x1, "n": 24},
        {"_type": "momentum", "delta_x": util.delta_x2, "n": 250},
    )
    normalized = normalize_wavepacket(wavepacket, idx, angle)
    return convert_wavepacket_to_basis(normalized, basis)


def load_two_point_normalized_nickel_wavepacket_momentum(
    band: int, angle: float = 0
) -> Wavepacket[
    Literal[12],
    Literal[12],
    MomentumBasisConfig[Literal[24], Literal[24], Literal[250]],
]:
    wavepacket = load_nickel_wavepacket(band)
    util = BasisConfigUtil(wavepacket["basis"])
    basis: MomentumBasisConfig[Literal[24], Literal[24], Literal[250]] = (
        {"_type": "momentum", "delta_x": util.delta_x0, "n": 24},
        {"_type": "momentum", "delta_x": util.delta_x1, "n": 24},
        {"_type": "momentum", "delta_x": util.delta_x2, "n": 250},
    )
    normalized = normalize_wavepacket_two_point(wavepacket, angle)
    wavepacket["basis"][0]["parent"]["n"] = 24
    wavepacket["basis"][1]["parent"]["n"] = 24
    return convert_wavepacket_to_basis(normalized, basis)


def generate_nickel_wavepacket_sho() -> None:
    def hamiltonian_generator(
        x: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    ) -> Hamiltonian[Any]:
        return generate_hamiltonian_sho(
            shape=(250, 250, 250),
            bloch_phase=x,
            resolution=(24, 24, 12),
        )

    save_bands = np.arange(20)

    wavepackets = generate_wavepacket(
        hamiltonian_generator,
        samples=(12, 12),
        save_bands=save_bands,
    )
    for k, wavepacket in zip(save_bands, wavepackets, strict=True):
        path = get_data_path(f"wavepacket_{k}.npy")
        save_wavepacket(path, wavepacket)
