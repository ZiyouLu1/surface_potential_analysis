from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
from surface_potential_analysis.axis.axis import (
    FundamentalBasis,
    FundamentalPositionBasis,
)
from surface_potential_analysis.axis.evenly_spaced_basis import (
    EvenlySpacedBasis,
)
from surface_potential_analysis.axis.stacked_axis import StackedBasis
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_position_basis,
)
from surface_potential_analysis.state_vector.conversion import (
    convert_state_vector_list_to_basis,
)
from surface_potential_analysis.state_vector.util import (
    get_most_localized_free_state_vectors,
)
from surface_potential_analysis.util.decorators import npy_cached
from surface_potential_analysis.wavepacket.localization import (
    localize_wavepacket_projection_many_band,
    localize_wavepacket_wannier90_many_band,
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

    from surface_potential_analysis.axis.axis import (
        FundamentalTransformedPositionBasis1d,
    )
    from surface_potential_analysis.axis.axis_like import BasisLike
    from surface_potential_analysis.axis.stacked_axis import (
        StackedBasisLike,
    )
    from surface_potential_analysis.operator.operator import SingleBasisOperator
    from surface_potential_analysis.state_vector.state_vector_list import (
        StateVectorList,
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


def _get_localized_wavepackets_cache(
    shape: tuple[_L0Inv], resolution: tuple[_L1Inv], n_bands: int = 0
) -> Path:
    return get_data_path(
        f"wavepacket/localized_wavepacket_{shape[0]}_{resolution[0]}_{n_bands}_bands.npy"
    )


def wavepacket3d_from_1d(
    wavepacket: WavepacketList[
        BasisLike[Any, Any],
        StackedBasisLike[*tuple[Any, ...]],
        StackedBasisLike[*tuple[Any, ...]],
    ]
) -> WavepacketList[Any, Any, Any]:
    converted = convert_state_vector_list_to_basis(
        wavepacket, stacked_basis_as_fundamental_position_basis(wavepacket["basis"][1])
    )
    return {
        "basis": StackedBasis(
            StackedBasis(
                converted["basis"][0][0],
                StackedBasis(
                    converted["basis"][0][1][0],
                    FundamentalBasis(1),
                    FundamentalBasis(1),
                ),
            ),
            StackedBasis(
                FundamentalPositionBasis(
                    np.array([converted["basis"][1][0].delta_x[0], 0, 0]),
                    converted["basis"][1][0].n,
                ),
                FundamentalPositionBasis(
                    np.array([0, converted["basis"][1][0].delta_x[0], 0]),
                    1,
                ),
                FundamentalPositionBasis(
                    np.array([0, 0, converted["basis"][1][0].delta_x[0]]),
                    1,
                ),
            ),
        ),
        "data": converted["data"],
    }


def state3d_from_1d(
    wavepacket: StateVectorList[Any, StackedBasisLike[*tuple[Any, ...]]]
) -> StateVectorList[Any, Any]:
    converted = convert_state_vector_list_to_basis(
        wavepacket, stacked_basis_as_fundamental_position_basis(wavepacket["basis"][1])
    )
    return {
        "basis": StackedBasis(
            converted["basis"][0],
            StackedBasis(
                FundamentalPositionBasis(
                    np.array([converted["basis"][1][0].delta_x[0], 0, 0]),
                    converted["basis"][1][0].n,
                ),
                FundamentalPositionBasis(
                    np.array([0, converted["basis"][1][0].delta_x[0], 0]),
                    1,
                ),
                FundamentalPositionBasis(
                    np.array([0, 0, converted["basis"][1][0].delta_x[0]]),
                    1,
                ),
            ),
        ),
        "data": converted["data"],
    }


@npy_cached(_get_localized_wavepackets_cache)
def get_localized_wavepackets_wannier_90(
    shape: tuple[_L0Inv], resolution: tuple[_L1Inv], n_bands: int = 0
) -> WavepacketList[
    StackedBasisLike[*tuple[FundamentalBasis[int], ...]],
    StackedBasisLike[EvenlySpacedBasis[_L0Inv, Literal[1], Literal[0]]],
    StackedBasisLike[FundamentalTransformedPositionBasis1d[_L1Inv]],
]:
    wavepackets = get_all_wavepackets(shape, resolution)
    wavepackets["data"] = (
        wavepackets["data"].reshape(wavepackets["basis"].shape)
        * np.exp(-1j * 2 * np.pi * np.random.rand(wavepackets["basis"].shape[0]))[
            :, np.newaxis
        ]
    ).reshape(-1)
    projections = get_most_localized_free_state_vectors(
        get_wavepacket_basis(wavepackets), (n_bands,)
    )
    return localize_wavepacket_wannier90_many_band(
        get_wavepackets(wavepackets, slice(n_bands)), projections
    )


def get_localized_wavepackets_projection(
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
    return localize_wavepacket_projection_many_band(
        get_wavepackets(wavepackets, slice(n_bands)), projections
    )
