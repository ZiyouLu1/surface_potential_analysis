from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from surface_potential_analysis.util.decorators import npy_cached
from surface_potential_analysis.wavepacket.localization import (
    localize_tightly_bound_wavepacket_two_point_max,
)
from surface_potential_analysis.wavepacket.wavepacket import (
    Wavepacket3dWith2dSamples,
    generate_wavepacket,
)

from .s2_hamiltonian import get_hamiltonian_hydrogen
from .surface_data import get_data_path

if TYPE_CHECKING:
    from pathlib import Path

    from surface_potential_analysis.axis.axis import (
        ExplicitAxis,
        TransformedPositionAxis,
    )
    from surface_potential_analysis.basis.basis import (
        Basis3d,
    )
    from surface_potential_analysis.operator import SingleBasisOperator

    _HydrogenCopperWavepacket = Wavepacket3dWith2dSamples[
        Literal[12],
        Literal[12],
        Basis3d[
            TransformedPositionAxis[Literal[23], Literal[23], Literal[3]],
            TransformedPositionAxis[Literal[23], Literal[23], Literal[3]],
            ExplicitAxis[Literal[250], Literal[14], Literal[3]],
        ],
    ]


@npy_cached(get_data_path("wavepacket/wavepacket_hydrogen.npy"), load_pickle=True)  # type: ignore[misc]
def get_all_wavepackets_hydrogen() -> list[_HydrogenCopperWavepacket]:
    def _hamiltonian_generator(
        bloch_fraction: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    ) -> SingleBasisOperator[Any]:
        return get_hamiltonian_hydrogen(
            shape=(46, 46, 250),
            bloch_fraction=bloch_fraction,
            resolution=(23, 23, 14),
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
def get_wavepacket_hydrogen(band: int) -> _HydrogenCopperWavepacket:
    return get_all_wavepackets_hydrogen()[band]


def get_two_point_normalized_wavepacket_hydrogen(
    band: int, offset: tuple[int, int] = (0, 0), angle: float = 0
) -> _HydrogenCopperWavepacket:
    wavepacket = get_wavepacket_hydrogen(band)
    return localize_tightly_bound_wavepacket_two_point_max(wavepacket, offset, angle)


def get_hydrogen_energy_difference(state_0: int, state_1: int) -> np.float_:
    wavepacket_0 = get_wavepacket_hydrogen(state_0)
    wavepacket_1 = get_wavepacket_hydrogen(state_1)
    return np.average(wavepacket_0["eigenvalues"]) - np.average(
        wavepacket_1["eigenvalues"]
    )
