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
from surface_potential_analysis.util.decorators import npy_cached
from surface_potential_analysis.wavepacket.localization import (
    Wannier90Options,
    get_localization_operator_wannier90,
)
from surface_potential_analysis.wavepacket.localization.localization_operator import (
    get_localized_wavepackets,
)
from surface_potential_analysis.wavepacket.wavepacket import (
    Wavepacket,
    WavepacketList,
    WavepacketWithEigenvaluesList,
    generate_wavepacket,
    get_average_eigenvalues,
    get_wavepacket,
    get_wavepackets,
)

from .s2_hamiltonian import get_hamiltonian_hydrogen
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

    _HCuBandsBasis = EvenlySpacedBasis[Literal[25], Literal[1], Literal[0]]
    _HCuWavepacketBasis = StackedBasisLike[
        TransformedPositionBasis[Literal[23], Literal[23], Literal[3]],
        TransformedPositionBasis[Literal[23], Literal[23], Literal[3]],
        ExplicitBasis[Literal[250], Literal[14], Literal[3]],
    ]
    _HCuSampleBasis = StackedBasisLike[
        FundamentalBasis[Literal[5]],
        FundamentalBasis[Literal[5]],
        FundamentalBasis[Literal[1]],
    ]

    _HydrogenCopperWavepacketList = WavepacketWithEigenvaluesList[
        _HCuBandsBasis,
        _HCuSampleBasis,
        _HCuWavepacketBasis,
    ]
    _HydrogenCopperWavepacket = Wavepacket[
        _HCuSampleBasis,
        _HCuWavepacketBasis,
    ]


@npy_cached(get_data_path("wavepacket/wavepacket_hydrogen.npy"), load_pickle=True)
def get_all_wavepackets_hydrogen() -> _HydrogenCopperWavepacketList:
    def _hamiltonian_generator(
        bloch_fraction: np.ndarray[tuple[Literal[3]], np.dtype[np.float64]],
    ) -> SingleBasisOperator[Any]:
        return get_hamiltonian_hydrogen(
            shape=(46, 46, 250),
            bloch_fraction=bloch_fraction,
            resolution=(23, 23, 14),
        )

    return generate_wavepacket(
        _hamiltonian_generator,
        list_basis=fundamental_stacked_basis_from_shape((5, 5, 1)),
        save_bands=EvenlySpacedBasis(25, 1, 0),
    )


def get_wavepacket_hydrogen(band: int) -> _HydrogenCopperWavepacket:
    return get_wavepacket(get_all_wavepackets_hydrogen(), band)


def _get_wavepacket_cache_wannier90_h(n_samples: int) -> Path:
    return get_data_path(f"wavepacket/localized_wavepacket_operator_{n_samples}.npy")


@npy_cached(_get_wavepacket_cache_wannier90_h, load_pickle=True)
def get_localization_operator_hydrogen(
    n_samples: int,
) -> LocalizationOperator[_HCuSampleBasis, FundamentalBasis[int], BasisLike[Any, Any]]:
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
) -> WavepacketList[FundamentalBasis[int], _HCuSampleBasis, _HCuWavepacketBasis]:
    wavepackets = get_all_wavepackets_hydrogen()
    operator = get_localization_operator_hydrogen(n_samples)
    return get_localized_wavepackets(
        get_wavepackets(wavepackets, slice(n_samples)), operator
    )


def get_hydrogen_energy_difference(state_0: int, state_1: int) -> np.float64:
    eigenvalues = get_average_eigenvalues(get_all_wavepackets_hydrogen())["data"]
    return eigenvalues[state_0] - eigenvalues[state_1]
