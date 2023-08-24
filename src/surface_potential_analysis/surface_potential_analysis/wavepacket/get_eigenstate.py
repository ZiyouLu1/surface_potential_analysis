from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from surface_potential_analysis.axis.axis_like import (
    AsTransformedAxis,
    AxisVector,
    AxisWithLengthLike,
)
from surface_potential_analysis.basis.conversion import (
    basis_as_fundamental_momentum_basis,
)
from surface_potential_analysis.basis.util import (
    BasisUtil,
    wrap_index_around_origin,
)
from surface_potential_analysis.state_vector.conversion import (
    convert_state_vector_to_position_basis,
)
from surface_potential_analysis.util.util import slice_along_axis
from surface_potential_analysis.wavepacket.conversion import convert_wavepacket_to_basis
from surface_potential_analysis.wavepacket.wavepacket import (
    Wavepacket,
    WavepacketWithEigenvalues,
    get_sample_basis,
    get_unfurled_basis,
)

if TYPE_CHECKING:
    from surface_potential_analysis._types import (
        SingleIndexLike,
        SingleStackedIndexLike,
    )
    from surface_potential_analysis.axis.axis import FundamentalPositionAxis
    from surface_potential_analysis.basis.basis import (
        AxisWithLengthBasis,
    )
    from surface_potential_analysis.state_vector.eigenstate_collection import Eigenstate
    from surface_potential_analysis.state_vector.state_vector import StateVector

    _B0Inv = TypeVar("_B0Inv", bound=AxisWithLengthBasis[Any])
    _DT = TypeVar("_DT", bound=np.dtype[Any])
    _NS0Inv = TypeVar("_NS0Inv", bound=int)
    _S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])
    _S1Inv = TypeVar("_S1Inv", bound=tuple[int | np.int_, ...])

    _NOInv = TypeVar("_NOInv", bound=int | np.int_)

_NF0Inv = TypeVar("_NF0Inv", bound=int)
_N0Inv = TypeVar("_N0Inv", bound=int)
_ND0Inv = TypeVar("_ND0Inv", bound=int)


def _pad_sample_axis(
    vectors: np.ndarray[_S0Inv, _DT], ns: _NS0Inv, offset: _NOInv, axis: int = -1
) -> np.ndarray[tuple[int, ...], _DT]:
    final_shape = np.array(vectors.shape)
    final_shape[axis] = ns * final_shape[axis]
    padded = np.zeros(final_shape, dtype=vectors.dtype)
    if offset < 0:
        # We could alternatively slice starting on zero
        # and roll at the end but this is worse for performance
        vectors = np.roll(vectors, -1, axis=axis)
    padded[slice_along_axis(slice(offset % ns, None, ns), axis)] = vectors

    return padded  # type: ignore[no-any-return]


def _truncate_sample_axis(
    vectors: np.ndarray[_S0Inv, _DT], ns: _NS0Inv, offset: _NOInv, axis: int = -1
) -> np.ndarray[tuple[int, ...], _DT]:
    truncated = vectors[slice_along_axis(slice(offset % ns, None, ns), axis)]
    if offset < 0:
        # We could alternatively roll before we take the slice
        # and slice(0, None, ns) but this is worse for performance
        truncated = np.roll(truncated, 1, axis=axis)
    return truncated  # type: ignore[no-any-return]


# ruff: noqa: D102
class WavepacketSampleAxis(
    AsTransformedAxis[_NF0Inv, _N0Inv], AxisWithLengthLike[_NF0Inv, _N0Inv, _ND0Inv]
):
    """Axis used to represent a single eigenstate from a wavepacket."""

    def __init__(  # noqa: PLR0913
        self,
        delta_x: AxisVector[_ND0Inv],
        *,
        n: int,
        fundamental_n: _NF0Inv,
        ns: int,
        offset: _NOInv,
    ) -> None:
        self._delta_x = delta_x
        self._n = n
        self._fundamental_n = fundamental_n
        self._offset = offset
        self._ns = ns
        assert self.fundamental_n >= self.n
        assert offset >= (1 - ns) // 2
        assert offset <= (ns - 1) // 2
        super().__init__()

    @property
    def delta_x(self) -> AxisVector[_ND0Inv]:
        return self._delta_x

    @property
    def n(self) -> _N0Inv:
        return self._n // self._ns  # type: ignore[return-value]

    @property
    def fundamental_n(self) -> _NF0Inv:
        return self._fundamental_n

    def __as_transformed__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_ | np.float_]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
        casted = vectors.astype(np.complex_, copy=False)
        return _pad_sample_axis(casted, self._ns, self._offset, axis)

    def __from_transformed__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_ | np.float_]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
        casted = vectors.astype(np.complex_, copy=False)
        return _truncate_sample_axis(casted, self._ns, self._offset, axis)


def _get_sampled_basis(
    basis: _B0Inv, shape: _S0Inv, offset: _S1Inv
) -> tuple[WavepacketSampleAxis[Any, Any, Any], ...]:
    unfurled = get_unfurled_basis(basis, shape)
    return tuple(
        WavepacketSampleAxis(
            parent.delta_x,
            fundamental_n=parent.fundamental_n,
            n=parent.n,
            ns=ns,
            offset=o,
        )
        for (parent, ns, o) in zip(unfurled, shape, offset, strict=True)
    )


def get_state_vector(
    wavepacket: Wavepacket[_S0Inv, _B0Inv], idx: SingleIndexLike
) -> StateVector[tuple[WavepacketSampleAxis[Any, Any, Any], ...]]:
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
    converted = convert_wavepacket_to_basis(
        wavepacket, basis_as_fundamental_momentum_basis(wavepacket["basis"])
    )
    util = BasisUtil(get_sample_basis(converted["basis"], converted["shape"]))
    idx = util.get_flat_index(idx) if isinstance(idx, tuple) else idx
    offset = util.get_stacked_index(idx)

    basis = _get_sampled_basis(converted["basis"], converted["shape"], offset)  # type: ignore[type-var]
    return {"basis": basis, "vector": converted["vectors"][idx]}


def get_bloch_state_vector(
    wavepacket: Wavepacket[_S0Inv, _B0Inv], idx: SingleIndexLike
) -> StateVector[_B0Inv]:
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
    util = BasisUtil(get_sample_basis(wavepacket["basis"], wavepacket["shape"]))
    idx = util.get_flat_index(idx) if isinstance(idx, tuple) else idx

    return {"basis": wavepacket["basis"], "vector": wavepacket["vectors"][idx]}


def get_all_eigenstates(
    wavepacket: WavepacketWithEigenvalues[_S0Inv, _B0Inv]
) -> list[Eigenstate[AxisWithLengthBasis[Any]]]:
    """
    Get the eigenstate of a given wavepacket at a specific index.

    Parameters
    ----------
    wavepacket : Wavepacket[_S0Inv, _B0Inv]

    Returns
    -------
    Eigenstate[_B0Inv].
    """
    converted = convert_wavepacket_to_basis(
        wavepacket, basis_as_fundamental_momentum_basis(wavepacket["basis"])
    )
    util = BasisUtil(get_sample_basis(converted["basis"], converted["shape"]))
    return [
        {
            "basis": _get_sampled_basis(converted["basis"], converted["shape"], offset),
            "vector": v,
            "eigenvalue": e,
        }
        for (v, e, *offset) in zip(
            converted["vectors"],
            wavepacket["eigenvalues"],
            *util.nk_points,
            strict=True,
        )
    ]


def get_all_states(
    wavepacket: Wavepacket[_S0Inv, _B0Inv]
) -> list[StateVector[AxisWithLengthBasis[Any]]]:
    """
    Get the eigenstate of a given wavepacket at a specific index.

    Parameters
    ----------
    wavepacket : Wavepacket[_S0Inv, _B0Inv]

    Returns
    -------
    Eigenstate[_B0Inv].
    """
    converted = convert_wavepacket_to_basis(
        wavepacket, basis_as_fundamental_momentum_basis(wavepacket["basis"])
    )
    util = BasisUtil(get_sample_basis(converted["basis"], converted["shape"]))
    return [
        {
            "basis": _get_sampled_basis(converted["basis"], converted["shape"], offset),
            "vector": v,
        }
        for (v, *offset) in zip(converted["vectors"], *util.nk_points, strict=True)
    ]


def get_tight_binding_state(
    wavepacket: Wavepacket[_S0Inv, _B0Inv],
    idx: SingleIndexLike = 0,
    origin: SingleIndexLike | None = None,
) -> StateVector[tuple[FundamentalPositionAxis[Any, Any], ...]]:
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
    StateVector[tuple[FundamentalPositionAxis[Any, Any], ...]]
        The localized state under the tight binding approximation
    """
    state_0 = convert_state_vector_to_position_basis(get_state_vector(wavepacket, idx))
    util = BasisUtil(state_0["basis"])
    if origin is None:
        idx_0: SingleStackedIndexLike = util.get_stacked_index(
            np.argmax(np.abs(state_0["vector"]), axis=-1)
        )
        origin = wrap_index_around_origin(wavepacket["basis"], idx_0, (0, 0, 0), (0, 1))
    # Under the tight binding approximation all state vectors are equal.
    # The corresponding localized state is just the state at some index
    # truncated to a single unit cell
    unit_cell_util = BasisUtil(wavepacket["basis"])
    relevant_idx = wrap_index_around_origin(
        wavepacket["basis"], unit_cell_util.fundamental_nx_points, origin, (0, 1)  # type: ignore[arg-type]
    )
    relevant_idx_flat = util.get_flat_index(relevant_idx, mode="wrap")
    out: StateVector[tuple[FundamentalPositionAxis[Any, Any], ...]] = {
        "basis": state_0["basis"],
        "vector": np.zeros_like(state_0["vector"]),
    }
    out["vector"][relevant_idx_flat] = state_0["vector"][relevant_idx_flat]
    return out
