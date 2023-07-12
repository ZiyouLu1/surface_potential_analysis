from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from surface_potential_analysis.axis.axis import FundamentalMomentumAxis3d
from surface_potential_analysis.basis.util import AxisWithLengthBasisUtil
from surface_potential_analysis.wavepacket import save_wavepacket
from surface_potential_analysis.wavepacket.conversion import convert_wavepacket_to_basis
from surface_potential_analysis.wavepacket.localization import (
    localize_tightly_bound_wavepacket_idx,
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
        MomentumAxis3d,
    )
    from surface_potential_analysis.basis.basis import (
        Basis3d,
        FundamentalMomentumBasis3d,
    )
    from surface_potential_analysis.operator import SingleBasisOperator3d


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
    util = AxisWithLengthBasisUtil(wavepacket["basis"])
    basis: FundamentalMomentumBasis3d[Literal[24], Literal[24], Literal[250]] = (
        FundamentalMomentumAxis3d(util.delta_x[0], 24),
        FundamentalMomentumAxis3d(util.delta_x[1], 24),
        FundamentalMomentumAxis3d(util.delta_x[2], 250),
    )
    normalized = localize_tightly_bound_wavepacket_idx(wavepacket, idx, angle)
    return convert_wavepacket_to_basis(normalized, basis)


def generate_wavepacket_sho() -> None:
    def hamiltonian_generator(
        x: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    ) -> SingleBasisOperator3d[Any]:
        return generate_hamiltonian_sho(
            shape=(48, 48, 250),
            bloch_fraction=x,
            resolution=(24, 24, 16),
        )

    save_bands = np.arange(20)

    wavepackets = generate_wavepacket(
        hamiltonian_generator, shape=(12, 12), save_bands=save_bands
    )
    for k, wavepacket in zip(save_bands, wavepackets, strict=True):
        path = get_data_path(f"wavepacket_{k}.npy")
        save_wavepacket(path, wavepacket)
