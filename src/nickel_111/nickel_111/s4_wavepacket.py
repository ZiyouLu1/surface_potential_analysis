from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from surface_potential_analysis.axis.axis import (
    FundamentalMomentumAxis3d,
    MomentumAxis3d,
)
from surface_potential_analysis.basis.util import Basis3dUtil
from surface_potential_analysis.wavepacket import save_wavepacket
from surface_potential_analysis.wavepacket.conversion import convert_wavepacket_to_basis
from surface_potential_analysis.wavepacket.localization import (
    localize_tightly_bound_wavepacket_idx,
    localize_tightly_bound_wavepacket_two_point_max,
)
from surface_potential_analysis.wavepacket.wavepacket import (
    Wavepacket3dWith2dSamples,
    generate_wavepacket,
    load_wavepacket,
)

from .s2_hamiltonian import generate_hamiltonian_sho
from .surface_data import get_data_path

if TYPE_CHECKING:
    from surface_potential_analysis._types import SingleIndexLike3d
    from surface_potential_analysis.axis.axis import (
        ExplicitAxis3d,
    )
    from surface_potential_analysis.basis.basis import (
        Basis3d,
        FundamentalMomentumBasis3d,
    )
    from surface_potential_analysis.operator import SingleBasisOperator3d

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
) -> Wavepacket3dWith2dSamples[
    Literal[12],
    Literal[12],
    Basis3d[
        MomentumAxis3d[Literal[24], Literal[24]],
        MomentumAxis3d[Literal[24], Literal[24]],
        ExplicitAxis3d[Literal[250], Literal[12]],
    ],
]:
    path = get_data_path(f"wavepacket_{band}.npy")
    wavepacket = load_wavepacket(path)
    wavepacket["basis"] = (
        MomentumAxis3d(wavepacket["basis"][0].delta_x, 24, 24),
        MomentumAxis3d(wavepacket["basis"][1].delta_x, 24, 24),
        wavepacket["basis"][2],
    )
    return wavepacket  # type: ignore[return-value]


def load_normalized_nickel_wavepacket_momentum(
    band: int, idx: SingleIndexLike3d = 0, angle: float = 0
) -> Wavepacket3dWith2dSamples[
    Literal[12],
    Literal[12],
    FundamentalMomentumBasis3d[Literal[24], Literal[24], Literal[250]],
]:
    wavepacket = load_nickel_wavepacket(band)
    util = Basis3dUtil(wavepacket["basis"])
    basis: FundamentalMomentumBasis3d[Literal[24], Literal[24], Literal[250]] = (
        FundamentalMomentumAxis3d(util.delta_x0, 24),
        FundamentalMomentumAxis3d(util.delta_x1, 24),
        FundamentalMomentumAxis3d(util.delta_x2, 250),
    )
    normalized = localize_tightly_bound_wavepacket_idx(wavepacket, idx, angle)
    return convert_wavepacket_to_basis(normalized, basis)


def load_two_point_normalized_nickel_wavepacket_momentum(
    band: int, offset: tuple[int, int] = (0, 0), angle: float = 0
) -> Wavepacket3dWith2dSamples[
    Literal[12],
    Literal[12],
    FundamentalMomentumBasis3d[Literal[24], Literal[24], Literal[250]],
]:
    wavepacket = load_nickel_wavepacket(band)
    util = Basis3dUtil(wavepacket["basis"])
    basis: FundamentalMomentumBasis3d[Literal[24], Literal[24], Literal[250]] = (
        FundamentalMomentumAxis3d(util.delta_x0, 24),
        FundamentalMomentumAxis3d(util.delta_x1, 24),
        FundamentalMomentumAxis3d(util.delta_x2, 250),
    )
    normalized = localize_tightly_bound_wavepacket_two_point_max(
        wavepacket, offset, angle
    )
    return convert_wavepacket_to_basis(normalized, basis)


def generate_nickel_wavepacket_sho() -> None:
    def hamiltonian_generator(
        bloch_fraction: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    ) -> SingleBasisOperator3d[Any]:
        return generate_hamiltonian_sho(
            shape=(250, 250, 250),
            bloch_fraction=bloch_fraction,
            resolution=(24, 24, 12),
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
