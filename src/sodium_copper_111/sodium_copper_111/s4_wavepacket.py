from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar

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

from .s2_hamiltonian import get_hamiltonian
from .surface_data import get_data_path

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from surface_potential_analysis.basis.basis import (
        FundamentalBasis,
        FundamentalTransformedPositionBasis1d,
    )
    from surface_potential_analysis.basis.basis_like import BasisLike
    from surface_potential_analysis.basis.stacked_basis import (
        StackedBasisLike,
    )
    from surface_potential_analysis.operator.operator import SingleBasisOperator
    from surface_potential_analysis.wavepacket.localization.localization_operator import (
        LocalizationOperator,
    )

    _L0Inv = TypeVar("_L0Inv", bound=int)
    _L1Inv = TypeVar("_L1Inv", bound=int)
    _L2Inv = TypeVar("_L2Inv", bound=int)

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
        bloch_fraction: np.ndarray[tuple[Literal[1]], np.dtype[np.float_]]
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
