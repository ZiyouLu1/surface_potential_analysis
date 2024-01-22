from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
from surface_potential_analysis.basis.evenly_spaced_basis import (
    EvenlySpacedBasis,
)
from surface_potential_analysis.basis.stacked_basis import StackedBasis
from surface_potential_analysis.state_vector.util import (
    get_most_localized_free_state_vectors,
)
from surface_potential_analysis.util.decorators import npy_cached
from surface_potential_analysis.wavepacket.localization import (
    Wannier90Options,
    get_localization_operator_wannier90,
    localize_wavepacket_projection,
)
from surface_potential_analysis.wavepacket.localization.localization_operator import (
    get_localized_wavepackets,
)
from surface_potential_analysis.wavepacket.wavepacket import (
    WavepacketList,
    WavepacketWithEigenvalues,
    WavepacketWithEigenvaluesList,
    generate_wavepacket,
    get_wavepacket_basis,
    get_wavepackets,
)

from .s2_hamiltonian import (
    get_hamiltonian,
    get_hamiltonian_2d,
    get_hamiltonian_flat,
    get_hamiltonian_flat_2d,
    get_hamiltonian_flat_lithium,
)
from .surface_data import get_data_path

if TYPE_CHECKING:
    from pathlib import Path

    from surface_potential_analysis.basis.basis import (
        FundamentalBasis,
        FundamentalTransformedPositionBasis,
        FundamentalTransformedPositionBasis1d,
        FundamentalTransformedPositionBasis2d,
    )
    from surface_potential_analysis.basis.basis_like import BasisLike
    from surface_potential_analysis.basis.stacked_basis import (
        StackedBasisLike,
    )
    from surface_potential_analysis.operator.operator import (
        SingleBasisOperator,
    )
    from surface_potential_analysis.wavepacket.localization.localization_operator import (
        LocalizationOperator,
    )

    _L0Inv = TypeVar("_L0Inv", bound=int)
    _L1Inv = TypeVar("_L1Inv", bound=int)
    _L2Inv = TypeVar("_L2Inv", bound=int)
    _L3Inv = TypeVar("_L3Inv", bound=int)

    _SodiumWavepacketList = WavepacketWithEigenvaluesList[
        EvenlySpacedBasis[_L0Inv, Literal[1], Literal[0]],
        StackedBasisLike[EvenlySpacedBasis[_L1Inv, Literal[1], Literal[0]]],
        StackedBasisLike[FundamentalTransformedPositionBasis1d[_L2Inv]],
    ]
    _SodiumWavepacket = WavepacketWithEigenvalues[
        StackedBasisLike[*tuple[FundamentalBasis[_L0Inv]]],
        StackedBasisLike[*tuple[FundamentalTransformedPositionBasis1d[_L1Inv]]],
    ]


def _get_all_wavepackets_cache(shape: tuple[_L0Inv], resolution: tuple[_L1Inv]) -> Path:
    return get_data_path(f"wavepacket/wavepacket_{shape[0]}_{resolution[0]}.npy")


@npy_cached(_get_all_wavepackets_cache, load_pickle=True)
def get_all_wavepackets(
    shape: tuple[_L0Inv], resolution: tuple[_L1Inv]
) -> _SodiumWavepacketList[_L1Inv, _L0Inv, _L1Inv]:
    def _hamiltonian_generator(
        bloch_fraction: np.ndarray[tuple[Literal[1]], np.dtype[np.float64]]
    ) -> SingleBasisOperator[
        StackedBasisLike[FundamentalTransformedPositionBasis1d[_L1Inv]]
    ]:
        return get_hamiltonian(shape=resolution, bloch_fraction=bloch_fraction)

    return generate_wavepacket(
        _hamiltonian_generator,
        list_basis=StackedBasis(EvenlySpacedBasis(shape[0], 1, 0)),
        save_bands=EvenlySpacedBasis(resolution[0], 1, 0),
    )


def get_wavepacket(
    shape: tuple[_L0Inv], resolution: tuple[_L1Inv], band: int = 0
) -> _SodiumWavepacket[_L0Inv, _L1Inv]:
    return get_all_wavepackets(shape, resolution)[band]  # type: ignore[no-any-return]


def _get_localization_operator_cache(
    shape: tuple[_L0Inv], resolution: tuple[_L1Inv], n_bands: int = 0
) -> Path:
    return get_data_path(
        f"wavepacket/localization_operator_{shape[0]}_{resolution[0]}_{n_bands}.npy"
    )


@npy_cached(_get_localization_operator_cache, load_pickle=True)
def get_localization_operator_sodium(
    shape: tuple[_L0Inv], resolution: tuple[_L1Inv], n_bands: int = 0
) -> LocalizationOperator[
    StackedBasisLike[EvenlySpacedBasis[_L0Inv, Literal[1], Literal[0]]],
    StackedBasisLike[*tuple[FundamentalBasis[int], ...]],
    BasisLike[Any, Any],
]:
    wavepackets = get_all_wavepackets(shape, resolution)
    projections = get_most_localized_free_state_vectors(
        get_wavepacket_basis(wavepackets), (n_bands,)
    )
    return get_localization_operator_wannier90(
        get_wavepackets(wavepackets, slice(n_bands)),
        options=Wannier90Options(projection=projections),
    )


def get_localized_wavepackets_wannier_90(
    shape: tuple[_L0Inv], resolution: tuple[_L1Inv], n_bands: int = 0
) -> WavepacketList[
    StackedBasisLike[*tuple[FundamentalBasis[int], ...]],
    StackedBasisLike[EvenlySpacedBasis[_L0Inv, Literal[1], Literal[0]]],
    StackedBasisLike[FundamentalTransformedPositionBasis1d[_L1Inv]],
]:
    wavepackets = get_all_wavepackets(shape, resolution)
    operator = get_localization_operator_sodium(shape, resolution, n_bands)
    return get_localized_wavepackets(
        get_wavepackets(wavepackets, slice(n_bands)), operator
    )


def get_projection_localized_wavepackets(
    shape: tuple[_L0Inv], resolution: tuple[_L1Inv], n_bands: int = 0
) -> WavepacketList[
    StackedBasisLike[*tuple[FundamentalBasis[int], ...]],
    StackedBasisLike[EvenlySpacedBasis[_L0Inv, Literal[1], Literal[0]]],
    StackedBasisLike[FundamentalTransformedPositionBasis1d[_L1Inv]],
]:
    wavepackets = get_all_wavepackets(shape, resolution)
    projections = get_most_localized_free_state_vectors(
        get_wavepacket_basis(wavepackets), (n_bands,)
    )
    return localize_wavepacket_projection(
        get_wavepackets(wavepackets, slice(n_bands)), projections
    )


def get_all_wavepackets_flat(
    shape: tuple[_L0Inv], resolution: tuple[_L1Inv]
) -> _SodiumWavepacketList[_L1Inv, _L0Inv, _L1Inv]:
    def _hamiltonian_generator(
        bloch_fraction: np.ndarray[tuple[Literal[1]], np.dtype[np.float64]]
    ) -> SingleBasisOperator[
        StackedBasisLike[FundamentalTransformedPositionBasis1d[_L1Inv]]
    ]:
        return get_hamiltonian_flat(shape=resolution, bloch_fraction=bloch_fraction)

    return generate_wavepacket(
        _hamiltonian_generator,
        list_basis=StackedBasis(EvenlySpacedBasis(shape[0], 1, 0)),
        save_bands=EvenlySpacedBasis(resolution[0], 1, 0),
    )


def _get_all_wavepackets_flat_2d_cache(
    shape: tuple[_L0Inv, _L1Inv], resolution: tuple[_L2Inv, _L3Inv]
) -> Path:
    return get_data_path(
        f"wavepacket/wavepacket_flat_{shape[0]}_{shape[1]}_{resolution[0]}_{resolution[1]}.npy"
    )


@npy_cached(_get_all_wavepackets_flat_2d_cache, load_pickle=True)
def get_all_wavepackets_flat_2d(
    shape: tuple[_L0Inv, _L1Inv], resolution: tuple[_L2Inv, _L3Inv]
) -> WavepacketWithEigenvaluesList[
    EvenlySpacedBasis[Any, Literal[1], Literal[0]],
    StackedBasisLike[
        EvenlySpacedBasis[_L0Inv, Literal[1], Literal[0]],
        EvenlySpacedBasis[_L1Inv, Literal[1], Literal[0]],
    ],
    StackedBasisLike[
        FundamentalTransformedPositionBasis2d[_L2Inv],
        FundamentalTransformedPositionBasis2d[_L3Inv],
    ],
]:
    def _hamiltonian_generator(
        bloch_fraction: np.ndarray[tuple[Literal[2]], np.dtype[np.float64]]
    ) -> SingleBasisOperator[
        StackedBasisLike[
            FundamentalTransformedPositionBasis[_L2Inv, Literal[2]],
            FundamentalTransformedPositionBasis[_L3Inv, Literal[2]],
        ],
    ]:
        return get_hamiltonian_flat_2d(shape=resolution, bloch_fraction=bloch_fraction)

    return generate_wavepacket(
        _hamiltonian_generator,
        list_basis=StackedBasis(
            EvenlySpacedBasis(shape[0], 1, 0), EvenlySpacedBasis(shape[1], 1, 0)
        ),
        save_bands=EvenlySpacedBasis(np.prod(resolution), 1, 0),
    )


def _get_all_wavepackets_2d_cache(
    shape: tuple[_L0Inv, _L1Inv], resolution: tuple[_L2Inv, _L3Inv]
) -> Path:
    return get_data_path(
        f"wavepacket/wavepacket_{shape[0]}_{shape[1]}_{resolution[0]}_{resolution[1]}.npy"
    )


@npy_cached(_get_all_wavepackets_2d_cache, load_pickle=True)
def get_all_wavepackets_2d(
    shape: tuple[_L0Inv, _L1Inv], resolution: tuple[_L2Inv, _L3Inv]
) -> WavepacketWithEigenvaluesList[
    EvenlySpacedBasis[Any, Literal[1], Literal[0]],
    StackedBasisLike[
        EvenlySpacedBasis[_L0Inv, Literal[1], Literal[0]],
        EvenlySpacedBasis[_L1Inv, Literal[1], Literal[0]],
    ],
    StackedBasisLike[
        FundamentalTransformedPositionBasis2d[_L2Inv],
        FundamentalTransformedPositionBasis2d[_L3Inv],
    ],
]:
    def _hamiltonian_generator(
        bloch_fraction: np.ndarray[tuple[Literal[2]], np.dtype[np.float64]]
    ) -> SingleBasisOperator[
        StackedBasisLike[
            FundamentalTransformedPositionBasis[_L2Inv, Literal[2]],
            FundamentalTransformedPositionBasis[_L3Inv, Literal[2]],
        ],
    ]:
        return get_hamiltonian_2d(shape=resolution, bloch_fraction=bloch_fraction)

    return generate_wavepacket(
        _hamiltonian_generator,
        list_basis=StackedBasis(
            EvenlySpacedBasis(shape[0], 1, 0), EvenlySpacedBasis(shape[1], 1, 0)
        ),
        save_bands=EvenlySpacedBasis(np.prod(resolution), 1, 0),
    )


def get_all_wavepackets_flat_lithium(
    shape: tuple[_L0Inv], resolution: tuple[_L1Inv]
) -> _SodiumWavepacketList[_L1Inv, _L0Inv, _L1Inv]:
    def _hamiltonian_generator(
        bloch_fraction: np.ndarray[tuple[Literal[1]], np.dtype[np.float64]]
    ) -> SingleBasisOperator[
        StackedBasisLike[FundamentalTransformedPositionBasis1d[_L1Inv]]
    ]:
        return get_hamiltonian_flat_lithium(
            shape=resolution, bloch_fraction=bloch_fraction
        )

    return generate_wavepacket(
        _hamiltonian_generator,
        list_basis=StackedBasis(EvenlySpacedBasis(shape[0], 1, 0)),
        save_bands=EvenlySpacedBasis(resolution[0], 1, 0),
    )
