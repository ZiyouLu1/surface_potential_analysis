from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from surface_potential_analysis.util.decorators import npy_cached
from surface_potential_analysis.wavepacket.wavepacket import (
    Wavepacket,
    generate_wavepacket,
)

from sodium_copper_111.s2_hamiltonian import get_hamiltonian
from sodium_copper_111.surface_data import get_data_path

if TYPE_CHECKING:
    from surface_potential_analysis.axis.axis import FundamentalMomentumAxis1d
    from surface_potential_analysis.hamiltonian.hamiltonian import Hamiltonian

    _SodiumWavepacket = Wavepacket[
        tuple[Literal[12]], tuple[FundamentalMomentumAxis1d[Literal[1000]]]
    ]


@npy_cached(get_data_path("wavepacket.npy"), allow_pickle=True)
def get_all_wavepackets() -> list[_SodiumWavepacket]:
    def hamiltonian_generator(
        bloch_fraction: np.ndarray[tuple[Literal[1]], np.dtype[np.float_]]
    ) -> Hamiltonian[tuple[FundamentalMomentumAxis1d[Literal[1000]]]]:
        return get_hamiltonian(shape=(1000,), bloch_fraction=bloch_fraction)

    save_bands = np.arange(20)

    return generate_wavepacket(
        hamiltonian_generator,
        shape=(12,),
        save_bands=save_bands,
    )


def get_wavepacket(band: int = 0) -> _SodiumWavepacket:
    return get_all_wavepackets()[band]


def get_two_band_wavepacket_eigenstate() -> (
    Wavepacket[tuple[Literal[24]], tuple[FundamentalMomentumAxis1d[Literal[1000]]]]
):
    wavepacket_0 = get_wavepacket(0)
    wavepacket_1 = get_wavepacket(1)

    energies = np.zeros(
        2 * wavepacket_0["energies"].shape[0], wavepacket_0["energies"].shape[1]
    )
    energies[:] = wavepacket_0["energies"]
    energies[:] = wavepacket_1["energies"]
    combined: Wavepacket[
        tuple[Literal[24]], tuple[FundamentalMomentumAxis1d[Literal[1000]]]
    ] = {
        "basis": (
            FundamentalMomentumAxis1d(
                wavepacket_0["basis"][0].delta_x * 2, wavepacket_0["basis"][0].n
            ),
        ),
        "energies": energies,
        "shape": (wavepacket_0["shape"][0] * 2,),
        "vectors": wavepacket_0["vectors"],
    }
    return combined
