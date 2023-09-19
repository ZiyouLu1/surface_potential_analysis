from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from surface_potential_analysis.axis.axis import (
    FundamentalBasis,
    TransformedPositionBasis,
)
from surface_potential_analysis.axis.stacked_axis import (
    StackedBasis,
)
from surface_potential_analysis.axis.util import BasisUtil
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_position_basis,
)
from surface_potential_analysis.wavepacket.wavepacket import get_unfurled_basis

if TYPE_CHECKING:
    from surface_potential_analysis.axis.axis_like import BasisLike
    from surface_potential_analysis.axis.stacked_axis import (
        StackedBasisLike,
    )
    from surface_potential_analysis.state_vector.state_vector import StateVector
    from surface_potential_analysis.state_vector.state_vector_list import (
        StateVectorList,
    )
    from surface_potential_analysis.types import IntLike_co, SingleFlatIndexLike
    from surface_potential_analysis.wavepacket.wavepacket import WavepacketBasis

    _B0Inv = TypeVar("_B0Inv", bound=BasisLike[Any, Any])
    _SB0Inv = TypeVar("_SB0Inv", bound=StackedBasisLike[*tuple[Any, ...]])
    _SBL0Inv = TypeVar("_SBL0Inv", bound=StackedBasisLike[*tuple[Any, ...]])
    _L0Inv = TypeVar("_L0Inv", bound=int)


def get_single_point_state_vector_excact(
    basis: _B0Inv, idx: SingleFlatIndexLike
) -> StateVector[_B0Inv]:
    data = np.zeros(basis.n, dtype=np.complex_)
    data[idx] = 1
    return {"basis": basis, "data": data}


def get_single_point_state_vectors(
    basis: WavepacketBasis[_SB0Inv, _SBL0Inv],
    n_bands: _L0Inv,
) -> StateVectorList[FundamentalBasis[_L0Inv], StackedBasis[Any]]:
    converted = stacked_basis_as_fundamental_position_basis(get_unfurled_basis(basis))
    data = np.zeros((n_bands, converted.n))
    for i, _n in enumerate(np.linspace(0, basis[1].n, n_bands, endpoint=False)):
        data[i, np.random.randint(0, converted.n)] = 1
    return {
        "basis": StackedBasis(FundamentalBasis(n_bands), converted),
        "data": data.reshape(-1),
    }


def get_most_localized_free_state_vectors(
    basis: WavepacketBasis[_SB0Inv, _SBL0Inv],
    shape: tuple[IntLike_co, ...],
) -> StateVectorList[
    StackedBasis[*tuple[FundamentalBasis[int], ...]],
    StackedBasis[*tuple[TransformedPositionBasis[Any, Any, Any], ...]],
]:
    """
    Get the most localized free states on the surface.

    A reasonable choice for the initial wavefunctions are the 'most localized'
    states we would find if we assume the potential is zero. In this case
    the states are evenly occupied upto some threshold frequency.

    Returns
    -------
    StateVectorList[StackedBasis[*tuple[FundamentalBasis[int], ...]], StackedBasis[Any]]
        The most localized states
    """
    n_bands = np.prod(np.asarray(shape))
    # TODO: properly deal with uneven sampled basis
    sample_basis = StackedBasis(
        *tuple(
            TransformedPositionBasis(
                b0.fundamental_n * b1.delta_x,
                # TODO: is this correct threshold k when not in 1D,
                # or do we need a bigger or smaller width than n_bands?
                b0.fundamental_n * n_bands,
                b0.fundamental_n * b1.fundamental_n,
            )
            for (b0, b1) in zip(basis[0], basis[1], strict=True)
        )
    )
    bands_basis = StackedBasis(*tuple(FundamentalBasis(int(n)) for n in shape))
    bands_util = BasisUtil(bands_basis)
    sample_fractions = BasisUtil(sample_basis).stacked_nx_points
    sample_fractions = tuple(
        f / n for (f, n) in zip(sample_fractions, sample_basis.shape, strict=True)
    )

    data = np.zeros((n_bands, sample_basis.n))
    data = np.exp(
        (-2j * np.pi)
        * np.tensordot(
            bands_util.stacked_nk_points,
            sample_fractions,
            axes=(0, 0),
        )
    )
    data /= np.sqrt(np.sum(np.abs(data) ** 2, axis=1))[:, np.newaxis]
    return {"basis": StackedBasis(bands_basis, sample_basis), "data": data.reshape(-1)}
