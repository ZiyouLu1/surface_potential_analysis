from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from surface_potential_analysis.basis.evenly_spaced_basis import EvenlySpacedBasis
from surface_potential_analysis.stacked_basis.build import (
    fundamental_stacked_basis_from_shape,
)
from surface_potential_analysis.util.decorators import npy_cached
from surface_potential_analysis.wavepacket.localization import (
    localize_tightly_bound_wavepacket_two_point_max,
)
from surface_potential_analysis.wavepacket.wavepacket import (
    Wavepacket,
    WavepacketWithEigenvaluesList,
    generate_wavepacket,
    get_average_eigenvalues,
    get_wavepacket,
)

from .s2_hamiltonian import get_hamiltonian_deuterium, get_hamiltonian_hydrogen
from .surface_data import get_data_path

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from surface_potential_analysis.basis.basis import (
        ExplicitBasis,
        FundamentalBasis,
        TransformedPositionBasis,
    )
    from surface_potential_analysis.basis.stacked_basis import StackedBasisLike
    from surface_potential_analysis.operator import SingleBasisOperator

    _HydrogenRutheniumWavepacketList = WavepacketWithEigenvaluesList[
        EvenlySpacedBasis[Literal[20], Literal[1], Literal[0]],
        StackedBasisLike[
            FundamentalBasis[Literal[12]],
            FundamentalBasis[Literal[12]],
            FundamentalBasis[Literal[1]],
        ],
        StackedBasisLike[
            TransformedPositionBasis[Literal[25], Literal[25], Literal[3]],
            TransformedPositionBasis[Literal[25], Literal[25], Literal[3]],
            ExplicitBasis[Literal[250], Literal[10], Literal[3]],
        ],
    ]

    _HydrogenRutheniumWavepacket = Wavepacket[
        StackedBasisLike[
            FundamentalBasis[Literal[12]],
            FundamentalBasis[Literal[12]],
            FundamentalBasis[Literal[1]],
        ],
        StackedBasisLike[
            TransformedPositionBasis[Literal[25], Literal[25], Literal[3]],
            TransformedPositionBasis[Literal[25], Literal[25], Literal[3]],
            ExplicitBasis[Literal[250], Literal[10], Literal[3]],
        ],
    ]

    _DeuteriumRutheniumWavepacketList = WavepacketWithEigenvaluesList[
        EvenlySpacedBasis[Literal[20], Literal[1], Literal[0]],
        StackedBasisLike[
            FundamentalBasis[Literal[12]],
            FundamentalBasis[Literal[12]],
            FundamentalBasis[Literal[1]],
        ],
        StackedBasisLike[
            TransformedPositionBasis[Literal[33], Literal[33], Literal[3]],
            TransformedPositionBasis[Literal[33], Literal[33], Literal[3]],
            ExplicitBasis[Literal[250], Literal[10], Literal[3]],
        ],
    ]

    _DeuteriumRutheniumWavepacket = Wavepacket[
        StackedBasisLike[
            FundamentalBasis[Literal[12]],
            FundamentalBasis[Literal[12]],
            FundamentalBasis[Literal[1]],
        ],
        StackedBasisLike[
            TransformedPositionBasis[Literal[33], Literal[33], Literal[3]],
            TransformedPositionBasis[Literal[33], Literal[33], Literal[3]],
            ExplicitBasis[Literal[250], Literal[10], Literal[3]],
        ],
    ]


@npy_cached(get_data_path("wavepacket/wavepacket_hydrogen.npy"), load_pickle=True)
def get_all_wavepackets_hydrogen() -> _HydrogenRutheniumWavepacketList:
    def _hamiltonian_generator(
        bloch_fraction: np.ndarray[tuple[Literal[3]], np.dtype[np.float64]],
    ) -> SingleBasisOperator[Any]:
        return get_hamiltonian_hydrogen(
            shape=(50, 50, 250),
            bloch_fraction=bloch_fraction,
            resolution=(25, 25, 10),
        )

    return generate_wavepacket(
        _hamiltonian_generator,
        list_basis=fundamental_stacked_basis_from_shape((12, 12, 1)),
        save_bands=EvenlySpacedBasis(20, 1, 0),
    )


def _get_wavepacket_cache_h(band: int) -> Path:
    return get_data_path(f"wavepacket/wavepacket_hydrogen_{band}.npy")


@npy_cached(_get_wavepacket_cache_h, load_pickle=True)
def get_wavepacket_hydrogen(band: int) -> _HydrogenRutheniumWavepacket:
    return get_wavepacket(get_all_wavepackets_hydrogen(), band)


def get_two_point_normalized_wavepacket_hydrogen(
    band: int, offset: tuple[int, int] = (0, 0), angle: float = 0
) -> _HydrogenRutheniumWavepacket:
    wavepacket = get_wavepacket_hydrogen(band)
    return localize_tightly_bound_wavepacket_two_point_max(wavepacket, offset, angle)


def get_hydrogen_energy_difference(state_0: int, state_1: int) -> np.float64:
    eigenvalues = get_average_eigenvalues(get_all_wavepackets_hydrogen())["data"]
    return eigenvalues[state_0] - eigenvalues[state_1]


@npy_cached(get_data_path("wavepacket/wavepacket_deuterium.npy"), load_pickle=True)
def get_all_wavepackets_deuterium() -> _DeuteriumRutheniumWavepacketList:
    def _hamiltonian_generator(
        bloch_fraction: np.ndarray[tuple[Literal[3]], np.dtype[np.float64]],
    ) -> SingleBasisOperator[Any]:
        return get_hamiltonian_deuterium(
            shape=(66, 66, 250),
            bloch_fraction=bloch_fraction,
            resolution=(33, 33, 10),
        )

    return generate_wavepacket(
        _hamiltonian_generator,
        list_basis=fundamental_stacked_basis_from_shape((12, 12, 1)),
        save_bands=EvenlySpacedBasis(20, 1, 0),
    )


def _get_wavepacket_cache_d(band: int) -> Path:
    return get_data_path(f"wavepacket/wavepacket_deuterium_{band}.npy")


@npy_cached(_get_wavepacket_cache_d, load_pickle=True)
def get_wavepacket_deuterium(band: int) -> _DeuteriumRutheniumWavepacket:
    return get_wavepacket(get_all_wavepackets_deuterium(), band)


def get_two_point_normalized_wavepacket_deuterium(
    band: int, offset: tuple[int, int] = (0, 0), angle: float = 0
) -> _DeuteriumRutheniumWavepacket:
    wavepacket = get_wavepacket_deuterium(band)
    return localize_tightly_bound_wavepacket_two_point_max(wavepacket, offset, angle)


def get_deuterium_energy_difference(state_0: int, state_1: int) -> np.float64:
    eigenvalues = get_average_eigenvalues(get_all_wavepackets_deuterium())["data"]
    return eigenvalues[state_0] - eigenvalues[state_1]
