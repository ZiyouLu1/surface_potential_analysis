from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from surface_potential_analysis.basis.basis import (
    FundamentalBasis,
)
from surface_potential_analysis.basis.evenly_spaced_basis import EvenlySpacedBasis
from surface_potential_analysis.basis.stacked_basis import (
    StackedBasis,
    StackedBasisLike,
)
from surface_potential_analysis.stacked_basis.build import (
    fundamental_stacked_basis_from_shape,
)
from surface_potential_analysis.state_vector.util import (
    get_most_localized_free_state_vectors,
)
from surface_potential_analysis.util.decorators import npy_cached
from surface_potential_analysis.wavepacket.localization import (
    Wannier90Options,
    get_localization_operator_wannier90,
    localize_single_point_projection,
    localize_tight_binding_projection,
    localize_tightly_bound_wavepacket_two_point_max,
    localize_wavepacket_projection,
)
from surface_potential_analysis.wavepacket.localization.localization_operator import (
    get_localized_wavepacket_hamiltonian,
    get_localized_wavepackets,
    get_wavepacket_hamiltonian,
)
from surface_potential_analysis.wavepacket.wavepacket import (
    BlochWavefunctionList,
    BlochWavefunctionListList,
    BlochWavefunctionListWithEigenvalues,
    BlochWavefunctionListWithEigenvaluesList,
    generate_wavepacket,
    get_average_eigenvalues,
    get_wavepacket_basis,
    get_wavepacket_with_eigenvalues,
    get_wavepackets,
    get_wavepackets_with_eigenvalues,
)

from .s2_hamiltonian import (
    get_hamiltonian,
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
    from surface_potential_analysis.operator.operator import (
        SingleBasisOperator,
    )
    from surface_potential_analysis.operator.operator_list import (
        DiagonalOperatorList,
        OperatorList,
    )
    from surface_potential_analysis.wavepacket.localization.localization_operator import (
        LocalizationOperator,
    )

    _HCuBandsBasis = EvenlySpacedBasis[Literal[25], Literal[1], Literal[0]]
    _HCuWavepacketBasis = StackedBasisLike[
        TransformedPositionBasis[Literal[21], Literal[21], Literal[3]],
        TransformedPositionBasis[Literal[21], Literal[21], Literal[3]],
        ExplicitBasis[Literal[250], Literal[15], Literal[3]],
    ]
    _HCuSampleBasis = StackedBasisLike[
        FundamentalBasis[Literal[5]],
        FundamentalBasis[Literal[5]],
        FundamentalBasis[Literal[1]],
    ]

    _HydrogenCopperWavepacketList = BlochWavefunctionListWithEigenvaluesList[
        _HCuBandsBasis,
        _HCuSampleBasis,
        _HCuWavepacketBasis,
    ]
    _HydrogenCopperWavepacket = BlochWavefunctionList[
        _HCuSampleBasis,
        _HCuWavepacketBasis,
    ]


@npy_cached(get_data_path("wavepacket/wavepacket_hydrogen.npy"), load_pickle=True)  # type: ignore[misc]
def get_all_wavepackets_hydrogen() -> _HydrogenCopperWavepacketList:
    def _hamiltonian_generator(
        bloch_fraction: np.ndarray[tuple[Literal[3]], np.dtype[np.float64]],
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


def get_wavepacket_hydrogen(
    band: int,
) -> BlochWavefunctionListWithEigenvalues[_HCuSampleBasis, _HCuWavepacketBasis]:
    return get_wavepacket_with_eigenvalues(get_all_wavepackets_hydrogen(), band)


def get_hamiltonian_hydrogen() -> (
    DiagonalOperatorList[_HCuBandsBasis, _HCuSampleBasis, _HCuSampleBasis]
):
    wavepackets = get_all_wavepackets_hydrogen()
    return get_wavepacket_hamiltonian(wavepackets)


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
        f"wavepacket/localized_wavepacket_sp_hydrogen_{band}_five_2_symmetric.npy"
    )


@npy_cached(_get_wavepacket_cache_single_point_h, load_pickle=True)
def get_single_point_projection_localized_wavepacket_hydrogen(
    band: int,
) -> _HydrogenCopperWavepacket:
    wavepacket = get_wavepacket_hydrogen(band)
    return localize_single_point_projection(wavepacket)


def get_projection_localized_wavepackets(
    sample_shape: tuple[int, int, int],
) -> BlochWavefunctionListList[
    StackedBasisLike[*tuple[FundamentalBasis[int], ...]],
    _HCuSampleBasis,
    _HCuWavepacketBasis,
]:
    n_samples = sample_shape[0] * sample_shape[1]
    wavepackets = get_all_wavepackets_hydrogen()
    projections = get_most_localized_free_state_vectors(
        get_wavepacket_basis(wavepackets), sample_shape
    )
    return localize_wavepacket_projection(
        get_wavepackets(wavepackets, slice(n_samples)), projections
    )


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
) -> BlochWavefunctionListList[
    FundamentalBasis[int], _HCuSampleBasis, _HCuWavepacketBasis
]:
    wavepackets = get_all_wavepackets_hydrogen()
    operator = get_localization_operator_hydrogen(n_samples)
    return get_localized_wavepackets(
        get_wavepackets(wavepackets, slice(n_samples)), operator
    )


def get_localized_hamiltonian_hydrogen(
    n_samples: int,
) -> OperatorList[_HCuSampleBasis, FundamentalBasis[int], FundamentalBasis[int]]:
    wavepackets = get_all_wavepackets_hydrogen()
    operator = get_localization_operator_hydrogen(n_samples)
    return get_localized_wavepacket_hamiltonian(
        get_wavepackets_with_eigenvalues(wavepackets, slice(n_samples)), operator
    )


def get_hydrogen_energy_difference(state_0: int, state_1: int) -> np.complex128:
    eigenvalues = get_average_eigenvalues(get_all_wavepackets_hydrogen())["data"]
    return eigenvalues[state_0] - eigenvalues[state_1]
