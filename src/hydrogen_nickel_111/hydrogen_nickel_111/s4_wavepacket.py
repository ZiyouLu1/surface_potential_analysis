from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from surface_potential_analysis.basis.basis import (
    FundamentalBasis,
)
from surface_potential_analysis.basis.evenly_spaced_basis import EvenlySpacedBasis
from surface_potential_analysis.basis.stacked_basis import StackedBasis
from surface_potential_analysis.stacked_basis.build import (
    fundamental_stacked_basis_from_shape,
)
from surface_potential_analysis.util.decorators import npy_cached, timed
from surface_potential_analysis.wavepacket.localization import (
    Wannier90Options,
    get_localization_operator_wannier90,
    get_localization_operator_wannier90_individual_bands,
    localize_tightly_bound_wavepacket_two_point_max,
)
from surface_potential_analysis.wavepacket.localization.localization_operator import (
    get_localized_wavepacket_hamiltonian,
    get_localized_wavepackets,
    get_wavepacket_hamiltonian,
)
from surface_potential_analysis.wavepacket.wavepacket import (
    Wavepacket,
    WavepacketList,
    WavepacketWithEigenvalues,
    WavepacketWithEigenvaluesList,
    generate_wavepacket,
    get_average_eigenvalues,
    get_wavepacket,
    get_wavepacket_with_eigenvalues,
    get_wavepackets,
    get_wavepackets_with_eigenvalues,
)

from .s2_hamiltonian import (
    get_hamiltonian_deuterium,
    get_hamiltonian_hydrogen_extrapolated,
)
from .surface_data import get_data_path

if TYPE_CHECKING:
    from pathlib import Path

    from surface_potential_analysis.basis.basis import (
        ExplicitBasis,
        TransformedPositionBasis,
    )
    from surface_potential_analysis.basis.basis_like import BasisLike
    from surface_potential_analysis.basis.stacked_basis import StackedBasisLike
    from surface_potential_analysis.operator import SingleBasisOperator
    from surface_potential_analysis.operator.operator import SingleBasisDiagonalOperator
    from surface_potential_analysis.operator.operator_list import OperatorList
    from surface_potential_analysis.wavepacket.localization.localization_operator import (
        LocalizationOperator,
    )

    _HNiBandsBasis = EvenlySpacedBasis[Literal[25], Literal[1], Literal[0]]
    _HNiSampleBasis = StackedBasisLike[
        FundamentalBasis[Literal[11]],
        FundamentalBasis[Literal[11]],
        FundamentalBasis[Literal[1]],
    ]
    _HNiWavepacketBasis = StackedBasisLike[
        TransformedPositionBasis[Literal[21], Literal[21], Literal[3]],
        TransformedPositionBasis[Literal[21], Literal[21], Literal[3]],
        ExplicitBasis[Literal[64], Literal[15], Literal[3]],
    ]

    _HydrogenNickelWavepacketList = WavepacketWithEigenvaluesList[
        _HNiBandsBasis,
        _HNiSampleBasis,
        _HNiWavepacketBasis,
    ]

    _HydrogenNickelWavepacketWithEigenvalues = WavepacketWithEigenvalues[
        _HNiSampleBasis,
        _HNiWavepacketBasis,
    ]
    _DeuteriumNickelWavepacketList = WavepacketWithEigenvaluesList[
        EvenlySpacedBasis[Literal[25], Literal[1], Literal[0]],
        StackedBasisLike[
            FundamentalBasis[Literal[12]],
            FundamentalBasis[Literal[12]],
            FundamentalBasis[Literal[1]],
        ],
        StackedBasisLike[
            TransformedPositionBasis[Literal[27], Literal[27], Literal[3]],
            TransformedPositionBasis[Literal[27], Literal[27], Literal[3]],
            ExplicitBasis[Literal[200], Literal[10], Literal[3]],
        ],
    ]
    _DeuteriumNickelWavepacket = Wavepacket[
        StackedBasisLike[
            FundamentalBasis[Literal[12]],
            FundamentalBasis[Literal[12]],
            FundamentalBasis[Literal[1]],
        ],
        StackedBasisLike[
            TransformedPositionBasis[Literal[27], Literal[27], Literal[3]],
            TransformedPositionBasis[Literal[27], Literal[27], Literal[3]],
            ExplicitBasis[Literal[200], Literal[10], Literal[3]],
        ],
    ]


@npy_cached(get_data_path("wavepacket/wavepacket_hydrogen.npy"), load_pickle=True)
def get_all_wavepackets_hydrogen() -> _HydrogenNickelWavepacketList:
    @timed
    def _hamiltonian_generator(
        bloch_fraction: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    ) -> SingleBasisOperator[Any]:
        return get_hamiltonian_hydrogen_extrapolated(
            shape=(250, 250, 64),
            bloch_fraction=bloch_fraction,
            resolution=(29, 29, 13),
        )

    return generate_wavepacket(
        _hamiltonian_generator,
        list_basis=fundamental_stacked_basis_from_shape((11, 11, 1)),
        save_bands=EvenlySpacedBasis(25, 1, 0),
    )


def get_wavepacket_hydrogen(band: int) -> _HydrogenNickelWavepacketWithEigenvalues:
    return get_wavepacket_with_eigenvalues(get_all_wavepackets_hydrogen(), band)


def get_wavepacket_hamiltonian_hydrogen(
    n_samples: int,
) -> SingleBasisDiagonalOperator[
    StackedBasisLike[BasisLike[Any, Any], _HNiSampleBasis],
]:
    wavepackets = get_all_wavepackets_hydrogen()
    return get_wavepacket_hamiltonian(
        get_wavepackets_with_eigenvalues(wavepackets, slice(n_samples))
    )


def _get_wavepacket_cache_wannier90_h(start: int, end: int) -> Path:
    return get_data_path(f"wavepacket/localized_wavepacket_operator_{start}_{end}.npy")


@npy_cached(_get_wavepacket_cache_wannier90_h, load_pickle=True)
def get_localization_operator_hydrogen(
    start: int,
    end: int,
) -> LocalizationOperator[_HNiSampleBasis, FundamentalBasis[int], BasisLike[Any, Any]]:
    wavepackets = get_all_wavepackets_hydrogen()
    return get_localization_operator_wannier90(
        get_wavepackets(wavepackets, slice(start, end)),
        options=Wannier90Options(
            num_iter=100000,
            ignore_axes=(2,),
            convergence_tolerance=1e-20,
            projection={"basis": StackedBasis(FundamentalBasis(end - start))},
        ),
    )


def get_wannier90_localized_wavepacket_hydrogen(
    start: int,
    end: int,
) -> WavepacketList[FundamentalBasis[int], _HNiSampleBasis, _HNiWavepacketBasis]:
    wavepackets = get_all_wavepackets_hydrogen()
    operator = get_localization_operator_hydrogen(start, end)
    return get_localized_wavepackets(
        get_wavepackets(wavepackets, slice(start, end)), operator
    )


def get_wannier90_localized_hamiltonian_hydrogen(
    start: int,
    end: int,
) -> OperatorList[_HNiSampleBasis, FundamentalBasis[int], FundamentalBasis[int]]:
    wavepackets = get_all_wavepackets_hydrogen()
    operator = get_localization_operator_hydrogen(start, end)
    return get_localized_wavepacket_hamiltonian(
        get_wavepackets_with_eigenvalues(wavepackets, slice(start, end)), operator
    )


def _get_wavepacket_cache_individual_band_wannier90_h(n_samples: int) -> Path:
    return get_data_path(
        f"wavepacket/localized_wavepacket_operator_individual_{n_samples}.npy"
    )


@npy_cached(_get_wavepacket_cache_individual_band_wannier90_h, load_pickle=True)
def get_localization_operator_individual_bands_hydrogen(
    n_samples: int,
) -> LocalizationOperator[_HNiSampleBasis, BasisLike[Any, Any], BasisLike[Any, Any]]:
    wavepackets = get_all_wavepackets_hydrogen()
    return get_localization_operator_wannier90_individual_bands(
        get_wavepackets(wavepackets, slice(n_samples)),
    )


def get_wannier90_localized_individual_bands_wavepacket_hydrogen(
    n_samples: int,
) -> WavepacketList[BasisLike[Any, Any], _HNiSampleBasis, _HNiWavepacketBasis]:
    wavepackets = get_all_wavepackets_hydrogen()
    operator = get_localization_operator_individual_bands_hydrogen(n_samples)
    return get_localized_wavepackets(
        get_wavepackets(wavepackets, slice(n_samples)), operator
    )


def get_localization_operator_split_hydrogen() -> (
    LocalizationOperator[_HNiSampleBasis, FundamentalBasis[int], BasisLike[Any, Any]]
):
    # We localize the vibrational modes seperately from the groundstate wavefunctions
    # to hopefully get the 'best of both worlds'
    low_bands = get_localization_operator_individual_bands_hydrogen(2)
    high_bands = get_localization_operator_hydrogen(2, 8)
    data = np.zeros((low_bands["basis"][0].n, 8, 8), dtype=np.complex_)
    data[:, 2:8, 2:8] = high_bands["data"].reshape(-1, 6, 6)
    data[:, :2, :2] = low_bands["data"].reshape(-1, 2, 2)
    return {
        "basis": StackedBasis(
            low_bands["basis"][0],
            StackedBasis(FundamentalBasis(8), FundamentalBasis(8)),
        ),
        "data": data.reshape(-1),
    }


def get_wannier90_localized_split_bands_wavepacket_hydrogen() -> (
    WavepacketList[FundamentalBasis[int], _HNiSampleBasis, _HNiWavepacketBasis]
):
    wavepackets = get_all_wavepackets_hydrogen()
    operator = get_localization_operator_split_hydrogen()
    return get_localized_wavepackets(get_wavepackets(wavepackets, slice(8)), operator)


def get_wannier90_localized_split_bands_hamiltonian_hydrogen() -> (
    OperatorList[_HNiSampleBasis, FundamentalBasis[int], FundamentalBasis[int]]
):
    wavepackets = get_all_wavepackets_hydrogen()
    operator = get_localization_operator_split_hydrogen()
    return get_localized_wavepacket_hamiltonian(
        get_wavepackets_with_eigenvalues(wavepackets, slice(8)), operator
    )


def get_hydrogen_energy_difference(state_0: int, state_1: int) -> np.float_:
    eigenvalues = get_average_eigenvalues(get_all_wavepackets_hydrogen())["data"]
    return eigenvalues[state_0] - eigenvalues[state_1]


@npy_cached(get_data_path("wavepacket/wavepacket_deuterium.npy"), load_pickle=True)
def get_all_wavepackets_deuterium() -> _DeuteriumNickelWavepacketList:
    def _hamiltonian_generator(
        bloch_fraction: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    ) -> SingleBasisOperator[Any]:
        return get_hamiltonian_deuterium(
            shape=(200, 200, 200),
            bloch_fraction=bloch_fraction,
            resolution=(27, 27, 10),
        )

    return generate_wavepacket(
        _hamiltonian_generator,
        list_basis=fundamental_stacked_basis_from_shape((12, 12, 1)),
        save_bands=EvenlySpacedBasis(25, 1, 0),
    )


def _get_wavepacket_cache_d(band: int) -> Path:
    return get_data_path(f"wavepacket/wavepacket_deuterium_{band}.npy")


@npy_cached(_get_wavepacket_cache_d, load_pickle=True)
def get_wavepacket_deuterium(band: int) -> _DeuteriumNickelWavepacket:
    return get_wavepacket(get_all_wavepackets_deuterium(), band)


def get_two_point_normalized_wavepacket_deuterium(
    band: int, offset: tuple[int, int] = (0, 0), angle: float = 0
) -> _DeuteriumNickelWavepacket:
    wavepacket = get_wavepacket_deuterium(band)
    return localize_tightly_bound_wavepacket_two_point_max(wavepacket, offset, angle)


def get_deuterium_energy_difference(state_0: int, state_1: int) -> np.float_:
    eigenvalues = get_average_eigenvalues(get_all_wavepackets_deuterium())["data"]
    return eigenvalues[state_0] - eigenvalues[state_1]
