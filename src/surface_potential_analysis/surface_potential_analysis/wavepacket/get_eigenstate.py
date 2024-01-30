from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np

from surface_potential_analysis.basis.basis import (
    FundamentalBasis,
    FundamentalTransformedPositionBasis,
)
from surface_potential_analysis.basis.evenly_spaced_basis import (
    EvenlySpacedTransformedPositionBasis,
)
from surface_potential_analysis.basis.stacked_basis import (
    StackedBasis,
)
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_basis,
    stacked_basis_as_fundamental_momentum_basis,
)
from surface_potential_analysis.stacked_basis.util import (
    wrap_index_around_origin,
)
from surface_potential_analysis.state_vector.conversion import (
    convert_state_vector_list_to_basis,
    convert_state_vector_to_position_basis,
)
from surface_potential_analysis.state_vector.state_vector_list import get_state_vector
from surface_potential_analysis.types import (
    IntLike_co,
)
from surface_potential_analysis.wavepacket.conversion import (
    convert_wavepacket_to_fundamental_momentum_basis,
)
from surface_potential_analysis.wavepacket.wavepacket import (
    Wavepacket,
    WavepacketBasis,
    WavepacketList,
    WavepacketWithEigenvalues,
    get_sample_basis,
    get_wavepacket_basis,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import FundamentalPositionBasis
    from surface_potential_analysis.basis.basis_like import (
        BasisLike,
        BasisWithLengthLike,
    )
    from surface_potential_analysis.basis.stacked_basis import (
        StackedBasisLike,
    )
    from surface_potential_analysis.state_vector.eigenstate_collection import Eigenstate
    from surface_potential_analysis.state_vector.state_vector import StateVector
    from surface_potential_analysis.state_vector.state_vector_list import (
        StateVectorList,
    )
    from surface_potential_analysis.types import (
        SingleIndexLike,
        SingleStackedIndexLike,
    )

    _SBL1 = TypeVar(
        "_SBL1",
        bound=StackedBasisLike[*tuple[Any, ...]],
    )
    _BL0 = TypeVar(
        "_BL0",
        bound=BasisWithLengthLike[Any, Any, Any],
    )
    _FB0 = TypeVar("_FB0", bound=FundamentalBasis[Any])
    _FTB0 = TypeVar("_FTB0", bound=FundamentalTransformedPositionBasis[Any, Any])
    _B0Inv = TypeVar("_B0Inv", bound=BasisLike[Any, Any])
    _SB0 = TypeVar("_SB0", bound=StackedBasisLike[*tuple[Any, ...]])
    _SB1 = TypeVar("_SB1", bound=StackedBasisLike[*tuple[Any, ...]])
    _B0 = TypeVar("_B0", bound=BasisLike[Any, Any])


def _get_sampled_basis(
    basis: WavepacketBasis[
        StackedBasisLike[*tuple[_B0, ...]], StackedBasisLike[*tuple[_BL0, ...]]
    ],
    offset: tuple[IntLike_co, ...],
) -> StackedBasisLike[
    *tuple[EvenlySpacedTransformedPositionBasis[Any, Any, Any, Any], ...]
]:
    return StackedBasis(
        *tuple(
            EvenlySpacedTransformedPositionBasis[Any, Any, Any, Any](
                state_ax.delta_x * list_ax.n,
                n=state_ax.n,
                step=list_ax.n,
                offset=wrap_index_around_origin(
                    StackedBasis(FundamentalBasis(list_ax.n)), (o,), 0
                )[0],
            )
            for (list_ax, state_ax, o) in zip(basis[0], basis[1], offset, strict=True)
        )
    )


def get_wavepacket_state_vector(
    wavepacket: Wavepacket[_SB0, _SB1], idx: SingleIndexLike
) -> StateVector[
    StackedBasisLike[
        *tuple[EvenlySpacedTransformedPositionBasis[Any, Any, Any, Any], ...]
    ]
]:
    """
    Get the eigenstate of a given wavepacket at a specific index.

    Parameters
    ----------
    wavepacket : Wavepacket[_S0Inv, _B0Inv]
    idx : SingleIndexLike

    Returns
    -------
    Eigenstate[_B0Inv].
    """
    converted = convert_wavepacket_to_fundamental_momentum_basis(
        wavepacket,
        list_basis=stacked_basis_as_fundamental_basis(wavepacket["basis"][0]),
    )
    util = BasisUtil(converted["basis"][0])
    idx = util.get_flat_index(idx) if isinstance(idx, tuple) else idx
    offset = util.get_stacked_index(idx)

    basis = _get_sampled_basis(converted["basis"], offset)
    return {
        "basis": basis,
        "data": converted["data"].reshape(converted["basis"].shape)[idx],
    }


def get_bloch_state_vector(
    wavepacket: Wavepacket[_SB0, _SBL1], idx: SingleIndexLike
) -> StateVector[_SBL1]:
    """
    Get the eigenstate of a given wavepacket at a specific index.

    Parameters
    ----------
    wavepacket : Wavepacket[_S0Inv, _B0Inv]
    idx : SingleIndexLike

    Returns
    -------
    Eigenstate[_B0Inv].
    """
    util = BasisUtil(wavepacket["basis"][0])
    idx = util.get_flat_index(idx) if isinstance(idx, tuple) else idx
    return get_state_vector(wavepacket, idx)


def get_all_eigenstates(
    wavepacket: WavepacketWithEigenvalues[_SB0, _SBL1],
) -> list[
    Eigenstate[
        StackedBasisLike[
            *tuple[EvenlySpacedTransformedPositionBasis[Any, Any, Any, Any], ...]
        ]
    ]
]:
    """
    Get the eigenstate of a given wavepacket at a specific index.

    Parameters
    ----------
    wavepacket : Wavepacket[_S0Inv, _B0Inv]

    Returns
    -------
    Eigenstate[_B0Inv].
    """
    converted = convert_wavepacket_to_fundamental_momentum_basis(
        wavepacket,
        list_basis=stacked_basis_as_fundamental_basis(wavepacket["basis"][0]),
    )
    util = BasisUtil(get_sample_basis(converted["basis"]))
    return [
        {
            "basis": _get_sampled_basis(
                converted["basis"], cast(tuple[int, ...], offset)
            ),
            "data": v,
            "eigenvalue": e,
        }
        for (v, e, *offset) in zip(
            converted["data"],
            wavepacket["eigenvalue"],
            *util.stacked_nk_points,
            strict=True,
        )
    ]


def get_all_wavepacket_states(
    wavepacket: Wavepacket[_SB0, _SBL1],
) -> list[StateVector[StackedBasisLike[*tuple[Any, ...]]]]:
    """
    Get the eigenstate of a given wavepacket at a specific index.

    Parameters
    ----------
    wavepacket : Wavepacket[_S0Inv, _B0Inv]

    Returns
    -------
    Eigenstate[_B0Inv].
    """
    converted = convert_wavepacket_to_fundamental_momentum_basis(
        wavepacket,
        list_basis=stacked_basis_as_fundamental_basis(wavepacket["basis"][0]),
    )
    util = BasisUtil(get_sample_basis(converted["basis"]))
    return [
        {
            "basis": _get_sampled_basis(
                converted["basis"], cast(tuple[IntLike_co, ...], offset)
            ),
            "data": v,
        }
        for (v, *offset) in zip(
            converted["data"].reshape(converted["basis"].shape),
            *util.stacked_nk_points,
            strict=True,
        )
    ]


def get_tight_binding_state(
    wavepacket: Wavepacket[_SB0, _SBL1],
    idx: SingleIndexLike = 0,
    origin: SingleIndexLike | None = None,
) -> StateVector[StackedBasisLike[*tuple[FundamentalPositionBasis[Any, Any], ...]]]:
    """
    Given a wavepacket, get the state corresponding to the eigenstate under the tight binding approximation.

    Parameters
    ----------
    wavepacket : Wavepacket[_S0Inv, _B0Inv]
        The initial wavepacket
    idx : SingleIndexLike, optional
        The index of the state vector to use as reference, by default 0
    origin : SingleIndexLike | None, optional
        The origin about which to produce the localized state, by default the maximum of the wavefunction

    Returns
    -------
    StateVector[tuple[FundamentalPositionBasis[Any, Any], ...]]
        The localized state under the tight binding approximation
    """
    state_0 = convert_state_vector_to_position_basis(
        get_wavepacket_state_vector(wavepacket, idx)
    )
    util = BasisUtil(state_0["basis"])
    if origin is None:
        idx_0: SingleStackedIndexLike = util.get_stacked_index(
            int(np.argmax(np.abs(state_0["data"]), axis=-1))
        )
        origin = wrap_index_around_origin(wavepacket["basis"], idx_0, (0, 0, 0), (0, 1))
    # Under the tight binding approximation all state vectors are equal.
    # The corresponding localized state is just the state at some index
    # truncated to a single unit cell
    unit_cell_util = BasisUtil(wavepacket["basis"])
    relevant_idx = wrap_index_around_origin(
        wavepacket["basis"],
        unit_cell_util.fundamental_stacked_nx_points,
        origin,
        (0, 1),  # type: ignore[arg-type]
    )
    relevant_idx_flat = util.get_flat_index(relevant_idx, mode="wrap")
    out: StateVector[
        StackedBasisLike[*tuple[FundamentalPositionBasis[Any, Any], ...]]
    ] = {
        "basis": state_0["basis"],
        "data": np.zeros_like(state_0["data"]),
    }
    out["data"][relevant_idx_flat] = state_0["data"][relevant_idx_flat]
    return out


def get_states_at_bloch_idx(
    wavepackets: WavepacketList[
        _B0Inv,
        StackedBasisLike[*tuple[_FB0, ...]],
        StackedBasisLike[*tuple[_FTB0, ...]],
    ],
    idx: SingleIndexLike,
) -> StateVectorList[
    _B0Inv,
    StackedBasisLike[
        *tuple[EvenlySpacedTransformedPositionBasis[Any, Any, Any, Any], ...]
    ],
]:
    """
    Get all wavepacket states at the given bloch index.

    Returns
    -------
    StateVectorList[
    _B0Inv,
    StackedBasisLike[
        *tuple[EvenlySpacedTransformedPositionBasis[Any, Any, Any, Any], ...]
    ],
    ]
    """
    util = BasisUtil(wavepackets["basis"][0][1])
    idx = util.get_flat_index(idx) if isinstance(idx, tuple) else idx
    offset = util.get_stacked_index(idx)
    # TODO: support _SB0Inv not in fundamental, and make it so we dont need to convert to momentum basis

    converted = convert_state_vector_list_to_basis(
        wavepackets,
        stacked_basis_as_fundamental_momentum_basis(wavepackets["basis"][1]),
    )
    return {
        "basis": StackedBasis(
            converted["basis"][0][0],
            _get_sampled_basis(get_wavepacket_basis(converted), offset),
        ),
        "data": converted["data"]
        .reshape(*converted["basis"][0].shape, -1)[:, idx]
        .reshape(-1),
    }
