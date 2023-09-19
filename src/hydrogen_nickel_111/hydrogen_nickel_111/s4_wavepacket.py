from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from surface_potential_analysis.stacked_basis.build import (
    fundamental_stacked_basis_from_shape,
)
from surface_potential_analysis.util.decorators import npy_cached, timed
from surface_potential_analysis.wavepacket.localization import (
    localize_single_point_projection,
    localize_tight_binding_projection,
    localize_tightly_bound_wavepacket_two_point_max,
    localize_wavepacket_wannier90,
)
from surface_potential_analysis.wavepacket.wavepacket import (
    Wavepacket,
    WavepacketWithEigenvalues,
    generate_wavepacket,
)

from .s2_hamiltonian import (
    get_hamiltonian_deuterium,
    get_hamiltonian_hydrogen,
    get_hamiltonian_hydrogen_extrapolated,
)
from .surface_data import get_data_path

if TYPE_CHECKING:
    from pathlib import Path

    from surface_potential_analysis.axis.axis import (
        ExplicitBasis,
        FundamentalBasis,
        TransformedPositionBasis,
    )
    from surface_potential_analysis.axis.stacked_axis import StackedBasisLike
    from surface_potential_analysis.operator import SingleBasisOperator

    _HydrogenNickelWavepacketWithEigenvaluesOld = WavepacketWithEigenvalues[
        tuple[
            FundamentalBasis[Literal[12]],
            FundamentalBasis[Literal[12]],
            FundamentalBasis[Literal[1]],
        ],
        StackedBasisLike[
            tuple[
                TransformedPositionBasis[Literal[29], Literal[29], Literal[3]],
                TransformedPositionBasis[Literal[29], Literal[29], Literal[3]],
                ExplicitBasis[Literal[250], Literal[12], Literal[3]],
            ]
        ],
    ]
    _HydrogenNickelWavepacket = Wavepacket[
        StackedBasisLike[
            FundamentalBasis[Literal[12]],
            FundamentalBasis[Literal[12]],
            FundamentalBasis[Literal[1]],
        ],
        StackedBasisLike[
            TransformedPositionBasis[Literal[29], Literal[29], Literal[3]],
            TransformedPositionBasis[Literal[29], Literal[29], Literal[3]],
            ExplicitBasis[Literal[250], Literal[13], Literal[3]],
        ],
    ]

    _HydrogenNickelWavepacketWithEigenvalues = WavepacketWithEigenvalues[
        StackedBasisLike[
            FundamentalBasis[Literal[12]],
            FundamentalBasis[Literal[12]],
            FundamentalBasis[Literal[1]],
        ],
        StackedBasisLike[
            TransformedPositionBasis[Literal[29], Literal[29], Literal[3]],
            TransformedPositionBasis[Literal[29], Literal[29], Literal[3]],
            ExplicitBasis[Literal[250], Literal[13], Literal[3]],
        ],
    ]
    _DeuteriumNickelWavepacketWithEigenvalues = WavepacketWithEigenvalues[
        tuple[
            FundamentalBasis[Literal[12]],
            FundamentalBasis[Literal[12]],
            FundamentalBasis[Literal[1]],
        ],
        StackedBasisLike[
            tuple[
                TransformedPositionBasis[Literal[27], Literal[27], Literal[3]],
                TransformedPositionBasis[Literal[27], Literal[27], Literal[3]],
                ExplicitBasis[Literal[200], Literal[10], Literal[3]],
            ]
        ],
    ]
    _DeuteriumNickelWavepacket = Wavepacket[
        tuple[
            FundamentalBasis[Literal[12]],
            FundamentalBasis[Literal[12]],
            FundamentalBasis[Literal[1]],
        ],
        StackedBasisLike[
            tuple[
                TransformedPositionBasis[Literal[27], Literal[27], Literal[3]],
                TransformedPositionBasis[Literal[27], Literal[27], Literal[3]],
                ExplicitBasis[Literal[200], Literal[10], Literal[3]],
            ]
        ],
    ]


@npy_cached(get_data_path("wavepacket/wavepacket_hydrogen.npy"), load_pickle=True)
def get_all_wavepackets_hydrogen_old() -> (
    list[_HydrogenNickelWavepacketWithEigenvaluesOld]
):
    @timed
    def _hamiltonian_generator(
        bloch_fraction: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    ) -> SingleBasisOperator[Any]:
        return get_hamiltonian_hydrogen(
            shape=(250, 250, 250),
            bloch_fraction=bloch_fraction,
            resolution=(29, 29, 12),
        )

    save_bands = np.arange(20)
    return generate_wavepacket(
        _hamiltonian_generator,
        list_basis=fundamental_stacked_basis_from_shape((12, 12, 1)),
        save_bands=save_bands,
    )


@npy_cached(get_data_path("wavepacket/wavepacket_hydrogen.npy"), load_pickle=True)
def get_all_wavepackets_hydrogen() -> list[_HydrogenNickelWavepacketWithEigenvalues]:
    @timed
    def _hamiltonian_generator(
        bloch_fraction: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    ) -> SingleBasisOperator[Any]:
        return get_hamiltonian_hydrogen_extrapolated(
            shape=(250, 250, 250),
            bloch_fraction=bloch_fraction,
            resolution=(29, 29, 13),
        )

    save_bands = np.arange(20)
    return generate_wavepacket(
        _hamiltonian_generator,
        list_basis=fundamental_stacked_basis_from_shape((11, 11, 1)),
        save_bands=save_bands,
    )


@npy_cached(
    get_data_path("wavepacket/wavepacket_hydrogen_flipped.npy"), load_pickle=True
)
def get_all_wavepackets_hydrogen_flipped() -> (
    list[_HydrogenNickelWavepacketWithEigenvalues]
):
    @timed
    def _hamiltonian_generator(
        bloch_fraction: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    ) -> SingleBasisOperator[Any]:
        return get_hamiltonian_hydrogen_extrapolated(
            shape=(250, 250, 250),
            bloch_fraction=bloch_fraction,
            resolution=(29, 29, 13),
        )

    save_bands = np.arange(20)
    return generate_wavepacket(
        _hamiltonian_generator,
        list_basis=fundamental_stacked_basis_from_shape((11, 11, 1)),
        save_bands=save_bands,
    )


def _get_wavepacket_cache_h(band: int) -> Path:
    return get_data_path(f"wavepacket/wavepacket_hydrogen_{band}.npy")


@npy_cached(_get_wavepacket_cache_h, load_pickle=True)
def get_wavepacket_hydrogen(band: int) -> _HydrogenNickelWavepacketWithEigenvalues:
    return get_all_wavepackets_hydrogen()[band]


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


def _get_wavepacket_cache_wannier90_h(band: int) -> Path:
    return get_data_path(f"wavepacket/localized_wavepacket_w90_hydrogen_{band}.npy")


@npy_cached(_get_wavepacket_cache_wannier90_h, load_pickle=True)
def get_wannier90_localized_wavepacket_hydrogen(
    band: int,
) -> _HydrogenNickelWavepacket:
    wavepacket = get_wavepacket_hydrogen(band)
    return localize_wavepacket_wannier90(wavepacket)


def get_hydrogen_energy_difference(state_0: int, state_1: int) -> np.float_:
    wavepacket_0 = get_wavepacket_hydrogen(state_0)
    wavepacket_1 = get_wavepacket_hydrogen(state_1)
    return np.average(wavepacket_0["eigenvalues"]) - np.average(
        wavepacket_1["eigenvalues"]
    )


@npy_cached(get_data_path("wavepacket/wavepacket_deuterium.npy"), load_pickle=True)
def get_all_wavepackets_deuterium() -> list[_DeuteriumNickelWavepacketWithEigenvalues]:
    def _hamiltonian_generator(
        bloch_fraction: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    ) -> SingleBasisOperator[Any]:
        return get_hamiltonian_deuterium(
            shape=(200, 200, 200),
            bloch_fraction=bloch_fraction,
            resolution=(27, 27, 10),
        )

    save_bands = np.arange(20)
    return generate_wavepacket(
        _hamiltonian_generator,
        list_basis=fundamental_stacked_basis_from_shape((12, 12, 1)),
        save_bands=save_bands,
    )


def _get_wavepacket_cache_d(band: int) -> Path:
    return get_data_path(f"wavepacket/wavepacket_deuterium_{band}.npy")


@npy_cached(_get_wavepacket_cache_d, load_pickle=True)
def get_wavepacket_deuterium(band: int) -> _DeuteriumNickelWavepacketWithEigenvalues:
    return get_all_wavepackets_deuterium()[band]


def get_two_point_normalized_wavepacket_deuterium(
    band: int, offset: tuple[int, int] = (0, 0), angle: float = 0
) -> _DeuteriumNickelWavepacket:
    wavepacket = get_wavepacket_deuterium(band)
    return localize_tightly_bound_wavepacket_two_point_max(wavepacket, offset, angle)


def get_deuterium_energy_difference(state_0: int, state_1: int) -> np.float_:
    wavepacket_0 = get_wavepacket_deuterium(state_0)
    wavepacket_1 = get_wavepacket_deuterium(state_1)
    return np.average(wavepacket_0["eigenvalues"]) - np.average(
        wavepacket_1["eigenvalues"]
    )
