from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from surface_potential_analysis.axis.axis import (
    ExplicitAxis3d,
    FundamentalMomentumAxis3d,
    MomentumAxis3d,
)
from surface_potential_analysis.basis.util import (
    Basis3dUtil,
)
from surface_potential_analysis.wavepacket.conversion import convert_wavepacket_to_basis
from surface_potential_analysis.wavepacket.normalization import normalize_wavepacket
from surface_potential_analysis.wavepacket.wavepacket import (
    Wavepacket3dWith2dSamples,
    generate_wavepacket,
    load_wavepacket,
    save_wavepacket,
)

from .s2_hamiltonian import (
    generate_hamiltonian_sho,
    generate_hamiltonian_sho_relaxed,
)
from .surface_data import get_data_path

if TYPE_CHECKING:
    from surface_potential_analysis._types import SingleIndexLike3d
    from surface_potential_analysis.basis.basis import (
        Basis3d,
        FundamentalMomentumBasis3d,
    )
    from surface_potential_analysis.hamiltonian.hamiltonian import Hamiltonian3d


def load_copper_wavepacket(
    band: int,
) -> Wavepacket3dWith2dSamples[
    Literal[12],
    Literal[12],
    Basis3d[
        MomentumAxis3d[Literal[24], Literal[24]],
        MomentumAxis3d[Literal[24], Literal[24]],
        ExplicitAxis3d[Literal[250], Literal[18]],
    ],
]:
    path = get_data_path(f"wavepacket_{band}.npy")
    wavepacket = load_wavepacket(path)
    wavepacket["basis"][0]["parent"]["n"] = 24
    wavepacket["basis"][1]["parent"]["n"] = 24
    return wavepacket


def load_normalized_copper_wavepacket_momentum(
    band: int, idx: SingleIndexLike3d = 0, angle: float = 0
) -> Wavepacket3dWith2dSamples[
    Literal[12],
    Literal[12],
    FundamentalMomentumBasis3d[Literal[24], Literal[24], Literal[250]],
]:
    wavepacket = load_copper_wavepacket(band)
    util = Basis3dUtil(wavepacket["basis"])
    basis: FundamentalMomentumBasis3d[Literal[24], Literal[24], Literal[250]] = (
        FundamentalMomentumAxis3d(util.delta_x0, 24),
        FundamentalMomentumAxis3d(util.delta_x1, 24),
        FundamentalMomentumAxis3d(util.delta_x2, 250),
    )
    normalized = normalize_wavepacket(wavepacket, idx, angle)
    return convert_wavepacket_to_basis(normalized, basis)


def generate_wavepacket_sho_relaxed() -> None:
    def hamiltonian_generator(
        x: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    ) -> Hamiltonian3d[Any]:
        return generate_hamiltonian_sho_relaxed(
            shape=(46, 46, 250),
            bloch_phase=x,
            resolution=(23, 23, 18),
        )

    save_bands = np.arange(20)

    wavepackets = generate_wavepacket(
        hamiltonian_generator,
        shape=(12, 12),
        save_bands=save_bands,
    )
    for k, wavepacket in zip(save_bands, wavepackets, strict=True):
        path = get_data_path(f"wavepacket_relaxed_{k}.npy")
        save_wavepacket(path, wavepacket)


def generate_wavepacket_sho() -> None:
    def hamiltonian_generator(
        x: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    ) -> Hamiltonian3d[Any]:
        return generate_hamiltonian_sho(
            shape=(46, 46, 250),
            bloch_phase=x,
            resolution=(23, 23, 18),
        )

    save_bands = np.arange(20)

    wavepackets = generate_wavepacket(
        hamiltonian_generator,
        shape=(12, 12),
        save_bands=save_bands,
    )
    for k, wavepacket in zip(save_bands, wavepackets, strict=True):
        path = get_data_path(f"wavepacket_{k}.npy")
        save_wavepacket(path, wavepacket)
