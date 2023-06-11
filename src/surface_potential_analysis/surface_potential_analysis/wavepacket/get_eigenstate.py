from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from surface_potential_analysis.axis.axis_like import AxisLike, AxisVector
from surface_potential_analysis.basis.conversion import (
    basis_as_fundamental_momentum_basis,
)
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.util.util import slice_along_axis
from surface_potential_analysis.wavepacket.conversion import convert_wavepacket_to_basis
from surface_potential_analysis.wavepacket.wavepacket import (
    Wavepacket,
    get_sample_basis,
    get_unfurled_basis,
)

if TYPE_CHECKING:
    from surface_potential_analysis._types import SingleIndexLike
    from surface_potential_analysis.basis.basis import Basis
    from surface_potential_analysis.state_vector.state_vector import StateVector

    _B0Inv = TypeVar("_B0Inv", bound=Basis[Any])
    _DT = TypeVar("_DT", bound=np.dtype[Any])
_NF0Inv = TypeVar("_NF0Inv", bound=int)
_N0Inv = TypeVar("_N0Inv", bound=int)
_ND0Inv = TypeVar("_ND0Inv", bound=int)
_NS0Inv = TypeVar("_NS0Inv", bound=int)
_S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])
_S1Inv = TypeVar("_S1Inv", bound=tuple[int | np.int_, ...])

_NOInv = TypeVar("_NOInv", bound=int | np.int_)


def _pad_sample_axis(
    vectors: np.ndarray[_S0Inv, _DT], ns: _NS0Inv, offset: _NOInv, axis: int = -1
) -> np.ndarray[tuple[int, ...], _DT]:
    final_shape = np.array(vectors.shape)
    final_shape[axis] = ns * final_shape[axis]
    padded = np.zeros(final_shape)

    shifted_vectors = np.fft.fftshift(vectors, axes=(axis,))
    start = offset - (1 - ns) // 2
    padded[slice_along_axis(slice(start, None, ns), axis)] = shifted_vectors
    return np.fft.ifftshift(padded, axes=(axis,))  # type: ignore[no-any-return]  # type: ignore[no-any-return]


def _truncate_sample_axis(
    vectors: np.ndarray[_S0Inv, _DT], ns: _NS0Inv, offset: _NOInv, axis: int = -1
) -> np.ndarray[tuple[int, ...], _DT]:
    shifted = np.fft.fftshift(vectors, axes=(axis,))
    start = offset - (1 - ns) // 2
    truncated = shifted[slice_along_axis(slice(start, None, ns), axis)]
    return np.fft.ifftshift(truncated, axes=(axis,))  # type: ignore[no-any-return]


class SampledAxis(AxisLike[_NF0Inv, _N0Inv, _ND0Inv]):
    # TODO: Not sure if it possible to do the conversion correctly
    def __init__(
        self,
        parent: AxisLike[_NF0Inv, int, _ND0Inv],
        ns: int,
        offset: _NOInv,
    ) -> None:
        self._parent = parent
        self._offset = offset
        self._ns = ns
        assert self.fundamental_n >= self.n
        assert offset >= 0
        assert offset < ns
        super().__init__()

    @property
    def delta_x(self) -> AxisVector[_ND0Inv]:
        return self._parent.delta_x

    @property
    def n(self) -> _N0Inv:
        return self._parent.n // self._ns  # type: ignore[return-value]

    @property
    def fundamental_n(self) -> _NF0Inv:
        return self._parent.fundamental_n

    def __from_fundamental__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_ | np.float_]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
        raise NotImplementedError
        transformed = self._parent.__from_fundamental__(vectors, axis)
        return _truncate_sample_axis(transformed, self._ns, self._offset, axis)

    def __into_fundamental__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_ | np.float_]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
        raise NotImplementedError
        padded = _pad_sample_axis(vectors, self._ns, self._offset, axis)
        return self._parent.__into_fundamental__(padded, axis)


class WavepacketSampleAxis(AxisLike[_NF0Inv, _N0Inv, _ND0Inv]):
    def __init__(
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

    def __from_fundamental__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_ | np.float_]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
        transformed = np.fft.fft(vectors, self.fundamental_n, axis=axis, norm="ortho")
        return _truncate_sample_axis(transformed, self._ns, self._offset, axis)  # type: ignore[no-any-return]

    def __into_fundamental__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_ | np.float_]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
        padded = _pad_sample_axis(vectors, self._ns, self._offset, axis)
        return np.fft.ifft(padded, self.fundamental_n, axis, norm="ortho")  # type: ignore[no-any-return]


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


def get_eigenstate(
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

    basis = _get_sampled_basis(converted["basis"], converted["shape"], offset)
    return {"basis": basis, "vector": converted["vectors"][idx]}


def get_eigenstates(
    wavepacket: Wavepacket[_S0Inv, _B0Inv]
) -> list[StateVector[Basis[Any]]]:
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
    util = BasisUtil(get_sample_basis(wavepacket["basis"], wavepacket["shape"]))
    return [
        {
            "basis": _get_sampled_basis(
                wavepacket["basis"], wavepacket["shape"], offset
            ),
            "vector": v,
        }
        for (v, *offset) in zip(converted["vectors"], *util.nk_points, strict=True)
    ]
