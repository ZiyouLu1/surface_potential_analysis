from __future__ import annotations

from typing import Literal, TypeVar

import numpy as np

from surface_potential_analysis.axis.axis_like import (
    AxisLike,
    AxisVector,
)
from surface_potential_analysis.util.interpolation import pad_ft_points

_NF0Inv = TypeVar("_NF0Inv", bound=int)
_N0Inv = TypeVar("_N0Inv", bound=int)
_ND0Inv = TypeVar("_ND0Inv", bound=int)
# ruff: noqa: D102


class ExplicitAxis(AxisLike[_NF0Inv, _N0Inv, _ND0Inv]):
    """An axis with vectors given as explicit states."""

    def __init__(
        self,
        delta_x: AxisVector[_ND0Inv],
        vectors: np.ndarray[tuple[_N0Inv, _NF0Inv], np.dtype[np.complex_]],
    ) -> None:
        self._delta_x = delta_x
        self._vectors = vectors
        super().__init__()

    @property
    def delta_x(self) -> AxisVector[_ND0Inv]:
        return self._delta_x

    @property
    def n(self) -> _N0Inv:
        return self.vectors.shape[0]  # type: ignore[no-any-return]

    @property
    def fundamental_n(self) -> _NF0Inv:
        return self.vectors.shape[1]  # type: ignore[no-any-return]

    @property
    def vectors(self) -> np.ndarray[tuple[_N0Inv, _NF0Inv], np.dtype[np.complex_]]:
        return self._vectors

    @classmethod
    def from_momentum_vectors(
        cls: type[ExplicitAxis[_NF0Inv, _N0Inv, _ND0Inv]],
        delta_x: AxisVector[_ND0Inv],
        vectors: np.ndarray[tuple[_N0Inv, _NF0Inv], np.dtype[np.complex_]],
    ) -> ExplicitAxis[_NF0Inv, _N0Inv, _ND0Inv]:
        vectors = np.fft.ifft(vectors, axis=1, norm="ortho")
        return cls(delta_x, vectors)


class ExplicitAxis1d(ExplicitAxis[_NF0Inv, _N0Inv, Literal[1]]):
    """An axis with vectors given as explicit states with a 1d basis vector."""


class ExplicitAxis2d(ExplicitAxis[_NF0Inv, _N0Inv, Literal[2]]):
    """An axis with vectors given as explicit states with a 2d basis vector."""


class ExplicitAxis3d(ExplicitAxis[_NF0Inv, _N0Inv, Literal[3]]):
    """An axis with vectors given as explicit states with a 3d basis vector."""


class FundamentalPositionAxis(AxisLike[_NF0Inv, _NF0Inv, _ND0Inv]):
    """A axis with vectors that are the fundamental position states."""

    def __init__(self, delta_x: AxisVector[_ND0Inv], n: _NF0Inv) -> None:
        self._delta_x = delta_x
        self._n = n
        super().__init__()

    @property
    def delta_x(self) -> AxisVector[_ND0Inv]:
        return self._delta_x

    @property
    def n(self) -> _NF0Inv:
        return self._n

    @property
    def fundamental_n(self) -> _NF0Inv:
        return self._n

    @property
    def vectors(self) -> np.ndarray[tuple[_NF0Inv, _NF0Inv], np.dtype[np.complex_]]:
        return np.eye(self.n, self.n)  # type: ignore[no-any-return]


class FundamentalPositionAxis1d(FundamentalPositionAxis[_NF0Inv, Literal[1]]):
    """A axis with vectors that are the fundamental position states with a 1d basis vector."""


class FundamentalPositionAxis2d(FundamentalPositionAxis[_NF0Inv, Literal[2]]):
    """A axis with vectors that are the fundamental position states with a 2d basis vector."""


class FundamentalPositionAxis3d(FundamentalPositionAxis[_NF0Inv, Literal[3]]):
    """A axis with vectors that are the fundamental position states with a 3d basis vector."""


class MomentumAxis(AxisLike[_NF0Inv, _N0Inv, _ND0Inv]):
    """A axis with vectors which are the n lowest frequency momentum states."""

    def __init__(
        self, delta_x: AxisVector[_ND0Inv], n: _N0Inv, fundamental_n: _NF0Inv
    ) -> None:
        self._delta_x = delta_x
        self._n = n
        self._fundamental_n = fundamental_n
        super().__init__()

    @property
    def delta_x(self) -> AxisVector[_ND0Inv]:
        return self._delta_x

    @property
    def n(self) -> _N0Inv:
        return self._n

    @property
    def fundamental_n(self) -> _NF0Inv:
        return self._fundamental_n

    @property
    def vectors(self) -> np.ndarray[tuple[_N0Inv, _NF0Inv], np.dtype[np.complex_]]:
        all_states_in_k = np.eye(self.fundamental_n, self.fundamental_n)
        all_states_in_x = np.fft.ifft(all_states_in_k, axis=1, norm="ortho")
        # pad_ft_points selects just the n lowest momentum states
        return pad_ft_points(all_states_in_x, s=(self.n,), axes=(0,))  # type: ignore[return-value]


class MomentumAxis1d(MomentumAxis[_NF0Inv, _N0Inv, Literal[1]]):
    """A axis with vectors which are the n lowest frequency momentum states with a 1d basis vector."""


class MomentumAxis2d(MomentumAxis[_NF0Inv, _N0Inv, Literal[2]]):
    """A axis with vectors which are the n lowest frequency momentum states with a 2d basis vector."""


class MomentumAxis3d(MomentumAxis[_NF0Inv, _N0Inv, Literal[3]]):
    """A axis with vectors which are the n lowest frequency momentum states with a 3d basis vector."""


class FundamentalMomentumAxis(MomentumAxis[_NF0Inv, _NF0Inv, _ND0Inv]):
    """An axis with vectors which are the fundamental momentum states."""

    def __init__(self, delta_x: AxisVector[_ND0Inv], n: _NF0Inv) -> None:
        super().__init__(delta_x, n, n)

    @property
    def vectors(self) -> np.ndarray[tuple[_NF0Inv, _NF0Inv], np.dtype[np.complex_]]:
        all_states_in_k = np.eye(self.fundamental_n, self.fundamental_n)
        return np.fft.ifft(all_states_in_k, axis=1, norm="ortho")  # type: ignore[no-any-return]


class FundamentalMomentumAxis1d(FundamentalMomentumAxis[_NF0Inv, Literal[1]]):
    """An axis with vectors which are the fundamental momentum states with a 1d basis vector."""


class FundamentalMomentumAxis2d(FundamentalMomentumAxis[_NF0Inv, Literal[2]]):
    """An axis with vectors which are the fundamental momentum states with a 2d basis vector."""


class FundamentalMomentumAxis3d(FundamentalMomentumAxis[_NF0Inv, Literal[3]]):
    """An axis with vectors which are the fundamental momentum states with a 3d basis vector."""
