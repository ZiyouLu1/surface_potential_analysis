from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np
from surface_potential_analysis.util.decorators import npy_cached
from surface_potential_analysis.wavepacket.localization import (
    localize_position_operator,
)
from surface_potential_analysis.wavepacket.wavepacket import (
    Wavepacket,
    generate_wavepacket,
)

from sodium_copper_111.s2_hamiltonian import get_hamiltonian
from sodium_copper_111.surface_data import get_data_path

if TYPE_CHECKING:
    from pathlib import Path

    from surface_potential_analysis.axis.axis import FundamentalMomentumAxis1d
    from surface_potential_analysis.operator.operator import SingleBasisOperator

    _L0Inv = TypeVar("_L0Inv", bound=int)
    _L1Inv = TypeVar("_L1Inv", bound=int)

    _SodiumWavepacket = Wavepacket[
        tuple[_L0Inv], tuple[FundamentalMomentumAxis1d[_L1Inv]]
    ]


def _get_all_wavepackets_cache(shape: tuple[_L0Inv], resolution: tuple[_L1Inv]) -> Path:
    return get_data_path(f"wavepacket/wavepacket_{shape[0]}_{resolution[0]}.npy")


@npy_cached(_get_all_wavepackets_cache, load_pickle=True)  # type: ignore[misc]
def get_all_wavepackets(
    shape: tuple[_L0Inv], resolution: tuple[_L1Inv]
) -> list[_SodiumWavepacket[_L0Inv, _L1Inv]]:
    def _hamiltonian_generator(
        bloch_fraction: np.ndarray[tuple[Literal[1]], np.dtype[np.float_]]
    ) -> SingleBasisOperator[tuple[FundamentalMomentumAxis1d[_L1Inv]]]:
        return get_hamiltonian(shape=resolution, bloch_fraction=bloch_fraction)

    save_bands = np.arange(99)
    return generate_wavepacket(
        _hamiltonian_generator,
        shape=shape,
        save_bands=save_bands,
    )


def get_wavepacket(
    shape: tuple[_L0Inv], resolution: tuple[_L1Inv], band: int = 0
) -> _SodiumWavepacket[_L0Inv, _L1Inv]:
    return get_all_wavepackets(shape, resolution)[band]  # type: ignore[no-any-return]


def _get_localized_wavepackets_cache(
    shape: tuple[_L0Inv], resolution: tuple[_L1Inv], band: int = 0
) -> Path:
    return get_data_path(
        f"wavepacket/localized_wavepacket_{band}_{shape[0]}_{resolution[0]}.npy"
    )


@npy_cached(_get_localized_wavepackets_cache, load_pickle=True)  # type: ignore[misc]
def get_localized_wavepackets(
    shape: tuple[_L0Inv], resolution: tuple[_L1Inv], band: int = 0
) -> list[_SodiumWavepacket[_L0Inv, _L1Inv]]:
    return localize_position_operator(get_wavepacket(shape, resolution, band))
