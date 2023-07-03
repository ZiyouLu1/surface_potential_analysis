from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from surface_potential_analysis.axis.axis import (
    ExplicitAxis,
    FundamentalMomentumAxis3d,
    MomentumAxis3d,
)
from surface_potential_analysis.axis.conversion import axis_as_orthonormal_axis
from surface_potential_analysis.basis.util import Basis3dUtil
from surface_potential_analysis.util.decorators import npy_cached
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

from .s2_hamiltonian import get_hamiltonian_deuterium, get_hamiltonian_hydrogen_sho
from .surface_data import get_data_path

if TYPE_CHECKING:
    from pathlib import Path

    from surface_potential_analysis._types import SingleIndexLike3d
    from surface_potential_analysis.axis.axis import ExplicitAxis3d, MomentumAxis
    from surface_potential_analysis.basis.basis import (
        Basis3d,
        FundamentalMomentumBasis3d,
    )
    from surface_potential_analysis.operator import SingleBasisOperator3d

    _HydrogenNickelWavepacket = Wavepacket3dWith2dSamples[
        Literal[12],
        Literal[12],
        Basis3d[
            MomentumAxis[Literal[24], Literal[24], Literal[3]],
            MomentumAxis[Literal[24], Literal[24], Literal[3]],
            ExplicitAxis[Literal[250], Literal[12], Literal[3]],
        ],
    ]

    _DeuteriumNickelWavepacket = Wavepacket3dWith2dSamples[
        Literal[12],
        Literal[12],
        Basis3d[
            MomentumAxis[Literal[25], Literal[25], Literal[3]],
            MomentumAxis[Literal[25], Literal[25], Literal[3]],
            ExplicitAxis[Literal[200], Literal[10], Literal[3]],
        ],
    ]

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
    Basis3d[
        FundamentalMomentumAxis3d[Literal[24]],
        FundamentalMomentumAxis3d[Literal[24]],
        ExplicitAxis3d[Literal[250], Literal[12]],
    ],
]:
    wavepacket = load_nickel_wavepacket(band)
    util = Basis3dUtil(wavepacket["basis"])
    basis: Basis3d[
        FundamentalMomentumAxis3d[Literal[24]],
        FundamentalMomentumAxis3d[Literal[24]],
        ExplicitAxis3d[Literal[250], Literal[12]],
    ] = (
        FundamentalMomentumAxis3d(util.delta_x0, 24),
        FundamentalMomentumAxis3d(util.delta_x1, 24),
        wavepacket["basis"][2],
    )
    return localize_tightly_bound_wavepacket_two_point_max(
        {
            "basis": basis,
            "energies": wavepacket["energies"],
            "shape": wavepacket["shape"],
            "vectors": wavepacket["vectors"],
        },
        offset,
        angle,
    )


def generate_nickel_wavepacket_sho() -> None:
    def hamiltonian_generator(
        bloch_fraction: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    ) -> SingleBasisOperator3d[Any]:
        return get_hamiltonian_hydrogen_sho(
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


@npy_cached(get_data_path("wavepacket/wavepacket_hydrogen.npy"), load_pickle=True)  # type: ignore[misc]
def get_all_wavepackets_hydrogen() -> list[_HydrogenNickelWavepacket]:
    def _hamiltonian_generator(
        bloch_fraction: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    ) -> SingleBasisOperator3d[Any]:
        return get_hamiltonian_hydrogen_sho(
            shape=(250, 250, 250),
            bloch_fraction=bloch_fraction,
            resolution=(24, 24, 12),
        )

    save_bands = np.arange(20)
    return generate_wavepacket(
        _hamiltonian_generator,
        shape=(12, 12, 1),
        save_bands=save_bands,
    )


def _get_wavepacket_cache_h(band: int) -> Path:
    return get_data_path(f"wavepacket/wavepacket_hydrogen_{band}.npy")


@npy_cached(_get_wavepacket_cache_h, load_pickle=True)
def get_wavepacket_hydrogen(band: int) -> _HydrogenNickelWavepacket:
    wavepacket = get_all_wavepackets_hydrogen()[band]
    # Work around for bug in cached wavepacket
    wavepacket["basis"] = (
        FundamentalMomentumAxis3d(
            wavepacket["basis"][0].delta_x, wavepacket["basis"][0].n
        ),
        FundamentalMomentumAxis3d(
            wavepacket["basis"][1].delta_x, wavepacket["basis"][1].n
        ),
        axis_as_orthonormal_axis(wavepacket["basis"][2]),
    )
    return wavepacket


def get_two_point_normalized_wavepacket_hydrogen(
    band: int, offset: tuple[int, int] = (0, 0), angle: float = 0
) -> _HydrogenNickelWavepacket:
    wavepacket = get_wavepacket_hydrogen(band)
    return localize_tightly_bound_wavepacket_two_point_max(wavepacket, offset, angle)


@npy_cached(get_data_path("wavepacket/wavepacket_deuterium.npy"), load_pickle=True)  # type: ignore[misc]
def get_all_wavepackets_deuterium() -> list[_DeuteriumNickelWavepacket]:
    def _hamiltonian_generator(
        bloch_fraction: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    ) -> SingleBasisOperator3d[Any]:
        return get_hamiltonian_deuterium(
            shape=(200, 200, 200),
            bloch_fraction=bloch_fraction,
            resolution=(25, 25, 10),
        )

    save_bands = np.arange(20)
    return generate_wavepacket(
        _hamiltonian_generator,
        shape=(12, 12, 1),
        save_bands=save_bands,
    )


def _get_wavepacket_cache_d(band: int) -> Path:
    return get_data_path(f"wavepacket/wavepacket_deuterium_{band}.npy")


@npy_cached(_get_wavepacket_cache_d, load_pickle=True)
def get_wavepacket_deuterium(band: int) -> _DeuteriumNickelWavepacket:
    return get_all_wavepackets_deuterium()[band]


def get_two_point_normalized_wavepacket_deuterium(
    band: int, offset: tuple[int, int] = (0, 0), angle: float = 0
) -> _DeuteriumNickelWavepacket:
    wavepacket = get_wavepacket_deuterium(band)
    return localize_tightly_bound_wavepacket_two_point_max(wavepacket, offset, angle)
