from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

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
    localize_single_point_projection,
    localize_tight_binding_projection,
    localize_tightly_bound_wavepacket_two_point_max,
)
from surface_potential_analysis.wavepacket.localization.localization_operator import (
    get_localized_wavepackets,
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
)

from .s2_hamiltonian import (
    get_hamiltonian_deuterium,
    get_hamiltonian_hydrogen_extrapolated,
)
from .surface_data import get_data_path

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from surface_potential_analysis.basis.basis import (
        ExplicitBasis,
        TransformedPositionBasis,
    )
    from surface_potential_analysis.basis.basis_like import BasisLike
    from surface_potential_analysis.basis.stacked_basis import StackedBasisLike
    from surface_potential_analysis.operator import SingleBasisOperator
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
        ExplicitBasis[Literal[250], Literal[15], Literal[3]],
    ]

    _HydrogenNickelWavepacketList = WavepacketWithEigenvaluesList[
        _HNiBandsBasis,
        _HNiSampleBasis,
        _HNiWavepacketBasis,
    ]
    _HydrogenNickelWavepacket = Wavepacket[
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
            shape=(250, 250, 250),
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


def get_two_point_localized_wavepacket_hydrogen(
    band: int, offset: tuple[int, int] = (0, 0), angle: float = 0
) -> _HydrogenNickelWavepacket:
    wavepacket = get_wavepacket_hydrogen(band)
    return localize_tightly_bound_wavepacket_two_point_max(wavepacket, offset, angle)


def _get_wavepacket_cache_tight_binding_h(band: int) -> Path:
    return get_data_path(f"wavepacket/localized_wavepacket_tb_hydrogen_{band}.npy")


@npy_cached(_get_wavepacket_cache_tight_binding_h, load_pickle=True)
def get_tight_binding_projection_localized_wavepacket_hydrogen(
    band: int,
) -> _HydrogenNickelWavepacket:
    wavepacket = get_wavepacket_hydrogen(band)
    return localize_tight_binding_projection(wavepacket)


def _get_wavepacket_cache_single_point_h(band: int) -> Path:
    return get_data_path(f"wavepacket/localized_wavepacket_sp_hydrogen_{band}.npy")


@npy_cached(_get_wavepacket_cache_single_point_h, load_pickle=True)
def get_single_point_projection_localized_wavepacket_hydrogen(
    band: int,
) -> _HydrogenNickelWavepacket:
    wavepacket = get_wavepacket_hydrogen(band)
    return localize_single_point_projection(wavepacket)


def _get_wavepacket_cache_wannier90_h(n_samples: int) -> Path:
    return get_data_path(f"wavepacket/localized_wavepacket_operator_{n_samples}.npy")


@npy_cached(_get_wavepacket_cache_wannier90_h, load_pickle=True)
def get_localization_operator_hydrogen(
    n_samples: int,
) -> LocalizationOperator[_HNiSampleBasis, FundamentalBasis[int], BasisLike[Any, Any]]:
    wavepackets = get_all_wavepackets_hydrogen()
    return get_localization_operator_wannier90(
        get_wavepackets(wavepackets, slice(n_samples)),
        options=Wannier90Options(
            num_iter=100000,
            projection={"basis": StackedBasis(FundamentalBasis(n_samples))},
        ),
    )


def get_wannier90_localized_wavepacket_hydrogen(
    n_samples: int,
) -> WavepacketList[FundamentalBasis[int], _HNiSampleBasis, _HNiWavepacketBasis]:
    wavepackets = get_all_wavepackets_hydrogen()
    operator = get_localization_operator_hydrogen(n_samples)
    return get_localized_wavepackets(
        get_wavepackets(wavepackets, slice(n_samples)), operator
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
