from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from surface_potential_analysis.axis.evenly_spaced_basis import EvenlySpacedBasis
from surface_potential_analysis.stacked_basis.build import (
    fundamental_stacked_basis_from_shape,
)
from surface_potential_analysis.util.decorators import npy_cached
from surface_potential_analysis.wavepacket.localization import (
    localize_single_point_projection,
    localize_tight_binding_projection,
    localize_tightly_bound_wavepacket_two_point_max,
    localize_wavepacket_wannier90_many_band,
)
from surface_potential_analysis.wavepacket.wavepacket import (
    Wavepacket,
    WavepacketWithEigenvaluesList,
    generate_wavepacket,
    get_average_eigenvalues,
    get_wavepacket,
)

from .s2_hamiltonian import (
    get_hamiltonian,
)
from .surface_data import get_data_path

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from surface_potential_analysis.axis.axis import (
        ExplicitBasis,
        FundamentalBasis,
        TransformedPositionBasis,
    )
    from surface_potential_analysis.axis.stacked_axis import (
        StackedBasisLike,
    )
    from surface_potential_analysis.operator.operator import SingleBasisOperator

    _HydrogenCopperWavepacketList = WavepacketWithEigenvaluesList[
        EvenlySpacedBasis[Literal[25], Literal[1], Literal[0]],
        StackedBasisLike[
            FundamentalBasis[Literal[5]],
            FundamentalBasis[Literal[5]],
            FundamentalBasis[Literal[1]],
        ],
        StackedBasisLike[
            TransformedPositionBasis[Literal[21], Literal[21], Literal[3]],
            TransformedPositionBasis[Literal[21], Literal[21], Literal[3]],
            ExplicitBasis[Literal[250], Literal[15], Literal[3]],
        ],
    ]
    _HydrogenCopperWavepacket = Wavepacket[
        StackedBasisLike[
            FundamentalBasis[Literal[5]],
            FundamentalBasis[Literal[5]],
            FundamentalBasis[Literal[1]],
        ],
        StackedBasisLike[
            TransformedPositionBasis[Literal[21], Literal[21], Literal[3]],
            TransformedPositionBasis[Literal[21], Literal[21], Literal[3]],
            ExplicitBasis[Literal[250], Literal[15], Literal[3]],
        ],
    ]


@npy_cached(get_data_path("wavepacket/wavepacket_hydrogen.npy"), load_pickle=True)  # type: ignore[misc]
def get_all_wavepackets_hydrogen() -> _HydrogenCopperWavepacketList:
    def _hamiltonian_generator(
        bloch_fraction: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    ) -> SingleBasisOperator[Any]:
        return get_hamiltonian(
            shape=(48, 48, 251),
            bloch_fraction=bloch_fraction,
            resolution=(22, 21, 15),
        )

    return generate_wavepacket(
        _hamiltonian_generator,
        list_basis=fundamental_stacked_basis_from_shape((5, 5, 1)),
        save_bands=EvenlySpacedBasis(25, 1, 0),
    )


def get_wavepacket_hydrogen(band: int) -> _HydrogenCopperWavepacket:
    return get_wavepacket(get_all_wavepackets_hydrogen(), band)


def get_two_point_localized_wavepacket_hydrogen(
    band: int, offset: tuple[int, int] = (0, 0), angle: float = 0
) -> _HydrogenCopperWavepacket:
    wavepacket = get_wavepacket_hydrogen(band)
    return localize_tightly_bound_wavepacket_two_point_max(wavepacket, offset, angle)


def _get_wavepacket_cache_tight_binding_h(band: int) -> Path:
    return get_data_path(f"wavepacket/localized_wavepacket_tb_hydrogen_{band}.npy")


@npy_cached(_get_wavepacket_cache_tight_binding_h, load_pickle=True)
def get_tight_binding_projection_localized_wavepacket_hydrogen(
    band: int,
) -> _HydrogenCopperWavepacket:
    wavepacket = get_wavepacket_hydrogen(band)
    return localize_tight_binding_projection(wavepacket)


def _get_wavepacket_cache_single_point_h(band: int) -> Path:
    return get_data_path(
        f"wavepacket/localized_wavepacket_sp_hydrogen_{band}_five_2_symm.npy"
    )


@npy_cached(_get_wavepacket_cache_single_point_h, load_pickle=True)
def get_single_point_projection_localized_wavepacket_hydrogen(
    band: int,
) -> _HydrogenCopperWavepacket:
    wavepacket = get_wavepacket_hydrogen(band)
    return localize_single_point_projection(wavepacket)


def _get_wavepacket_cache_wannier90_h(band: int) -> Path:
    return get_data_path(f"wavepacket/localized_wavepacket_w90_hydrogen_{band}.npy")


@npy_cached(_get_wavepacket_cache_wannier90_h, load_pickle=True)
def get_wannier90_localized_wavepacket_hydrogen(
    band: int,
) -> _HydrogenCopperWavepacket:
    wavepacket = get_wavepacket_hydrogen(band)
    return localize_wavepacket_wannier90_many_band(wavepacket, projections)


def get_hydrogen_energy_difference(state_0: int, state_1: int) -> np.float_:
    eigenvalues = get_average_eigenvalues(get_all_wavepackets_hydrogen())["data"]
    return eigenvalues[state_0] - eigenvalues[state_1]
