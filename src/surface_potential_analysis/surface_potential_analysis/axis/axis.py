from __future__ import annotations

from typing import Literal, TypeVar

import numpy as np

from surface_potential_analysis.axis.axis_like import (
    AsFundamentalAxis,
    AsTransformedAxis,
    AxisLike,
    AxisVector,
    AxisWithLengthLike,
)
from surface_potential_analysis.util.interpolation import pad_ft_points

_NF0_co = TypeVar("_NF0_co", bound=int, covariant=True)
_N0_co = TypeVar("_N0_co", bound=int, covariant=True)

_NF0Inv = TypeVar("_NF0Inv", bound=int)
_N0Inv = TypeVar("_N0Inv", bound=int)
_ND0Inv = TypeVar("_ND0Inv", bound=int)

_S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])
# ruff: noqa: D102


class ExplicitAxis(AxisWithLengthLike[_NF0_co, _N0_co, _ND0Inv]):
    """An axis with vectors given as explicit states."""

    def __init__(
        self,
        delta_x: AxisVector[_ND0Inv],
        vectors: np.ndarray[tuple[_N0_co, _NF0_co], np.dtype[np.complex_]],
    ) -> None:
        self._delta_x = delta_x
        self._vectors = vectors
        super().__init__()

    @property
    def delta_x(self) -> AxisVector[_ND0Inv]:
        return self._delta_x

    @property
    def n(self) -> _N0_co:
        return self.vectors.shape[0]  # type: ignore[no-any-return]

    @property
    def fundamental_n(self) -> _NF0_co:
        return self.vectors.shape[1]  # type: ignore[no-any-return]

    @property
    def vectors(self) -> np.ndarray[tuple[_N0_co, _NF0_co], np.dtype[np.complex_]]:
        return self._vectors

    def __into_fundamental__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_ | np.float_]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
        transformed = np.tensordot(vectors, self.vectors, axes=([axis], [0]))
        return np.moveaxis(transformed, -1, axis)  # type: ignore[no-any-return]

    @classmethod
    def from_momentum_vectors(
        cls: type[ExplicitAxis[_NF0_co, _N0_co, _ND0Inv]],
        delta_x: AxisVector[_ND0Inv],
        vectors: np.ndarray[tuple[_N0_co, _NF0_co], np.dtype[np.complex_]],
    ) -> ExplicitAxis[_NF0_co, _N0_co, _ND0Inv]:
        vectors = np.fft.ifft(vectors, axis=1, norm="ortho")
        return cls(delta_x, vectors)


class ExplicitAxis1d(ExplicitAxis[_NF0Inv, _N0Inv, Literal[1]]):
    """An axis with vectors given as explicit states with a 1d basis vector."""


class ExplicitAxis2d(ExplicitAxis[_NF0Inv, _N0Inv, Literal[2]]):
    """An axis with vectors given as explicit states with a 2d basis vector."""


class ExplicitAxis3d(ExplicitAxis[_NF0Inv, _N0Inv, Literal[3]]):
    """An axis with vectors given as explicit states with a 3d basis vector."""


class FundamentalAxis(AsFundamentalAxis[_NF0_co, _NF0_co], AxisLike[_NF0_co, _NF0_co]):
    """A axis with vectors that are the fundamental position states."""

    def __init__(self, n: _NF0_co) -> None:
        self._n = n
        super().__init__()

    @property
    def n(self) -> _NF0_co:
        return self._n

    @property
    def fundamental_n(self) -> _NF0_co:
        return self._n

    def __as_fundamental__(  # type: ignore[override]
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_ | np.float_]],
        axis: int = -1,
    ) -> np.ndarray[_S0Inv, np.dtype[np.complex_]]:
        return vectors.astype(np.complex_, copy=False)  # type: ignore[no-any-return]

    def __from_fundamental__(  # type: ignore[override]
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_ | np.float_]],
        axis: int = -1,
    ) -> np.ndarray[_S0Inv, np.dtype[np.complex_]]:
        return vectors.astype(np.complex_, copy=False)  # type: ignore[no-any-return]


class FundamentalPositionAxis(
    FundamentalAxis[_NF0_co], AxisWithLengthLike[_NF0_co, _NF0_co, _ND0Inv]
):
    """A axis with vectors that are the fundamental position states."""

    def __init__(self, delta_x: AxisVector[_ND0Inv], n: _NF0_co) -> None:
        self._delta_x = delta_x
        super().__init__(n)

    @property
    def delta_x(self) -> AxisVector[_ND0Inv]:
        return self._delta_x


class FundamentalPositionAxis1d(FundamentalPositionAxis[_NF0Inv, Literal[1]]):
    """A axis with vectors that are the fundamental position states with a 1d basis vector."""


class FundamentalPositionAxis2d(FundamentalPositionAxis[_NF0Inv, Literal[2]]):
    """A axis with vectors that are the fundamental position states with a 2d basis vector."""


class FundamentalPositionAxis3d(FundamentalPositionAxis[_NF0Inv, Literal[3]]):
    """A axis with vectors that are the fundamental position states with a 3d basis vector."""


class TransformedPositionAxis(
    AsTransformedAxis[_NF0_co, _NF0_co],
    AxisWithLengthLike[_NF0_co, _N0_co, _ND0Inv],
):
    """A axis with vectors which are the n lowest frequency momentum states."""

    def __init__(
        self, delta_x: AxisVector[_ND0Inv], n: _N0_co, fundamental_n: _NF0_co
    ) -> None:
        self._delta_x = delta_x
        self._n = n
        self._fundamental_n = fundamental_n
        assert self._fundamental_n >= self.n
        super().__init__()

    @property
    def delta_x(self) -> AxisVector[_ND0Inv]:
        return self._delta_x

    @property
    def n(self) -> _N0_co:
        return self._n

    @property
    def fundamental_n(self) -> _NF0_co:
        return self._fundamental_n

    def __as_transformed__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_ | np.float_]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
        casted = vectors.astype(np.complex_, copy=False)
        return pad_ft_points(casted, s=(self.fundamental_n,), axes=(axis,))

    def __from_transformed__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_ | np.float_]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
        casted = vectors.astype(np.complex_, copy=False)
        return pad_ft_points(casted, s=(self.n,), axes=(axis,))


class TransformedPositionAxis1d(TransformedPositionAxis[_NF0Inv, _N0Inv, Literal[1]]):
    """A axis with vectors which are the n lowest frequency momentum states with a 1d basis vector."""


class TransformedPositionAxis2d(TransformedPositionAxis[_NF0Inv, _N0Inv, Literal[2]]):
    """A axis with vectors which are the n lowest frequency momentum states with a 2d basis vector."""


class TransformedPositionAxis3d(TransformedPositionAxis[_NF0Inv, _N0Inv, Literal[3]]):
    """A axis with vectors which are the n lowest frequency momentum states with a 3d basis vector."""


class FundamentalTransformedPositionAxis(
    TransformedPositionAxis[_NF0_co, _NF0_co, _ND0Inv]
):
    """An axis with vectors which are the fundamental momentum states."""

    def __init__(self, delta_x: AxisVector[_ND0Inv], n: _NF0_co) -> None:
        super().__init__(delta_x, n, n)

    @property
    def vectors(self) -> np.ndarray[tuple[_NF0_co, _NF0_co], np.dtype[np.complex_]]:
        all_states_in_k = np.eye(self.fundamental_n, self.fundamental_n)
        return np.fft.ifft(all_states_in_k, axis=1, norm="ortho")  # type: ignore[no-any-return]


class FundamentalTransformedPositionAxis1d(
    FundamentalTransformedPositionAxis[_NF0Inv, Literal[1]]
):
    """An axis with vectors which are the fundamental momentum states with a 1d basis vector."""


class FundamentalTransformedPositionAxis2d(
    FundamentalTransformedPositionAxis[_NF0Inv, Literal[2]]
):
    """An axis with vectors which are the fundamental momentum states with a 2d basis vector."""


class FundamentalTransformedPositionAxis3d(
    FundamentalTransformedPositionAxis[_NF0Inv, Literal[3]]
):
    """An axis with vectors which are the fundamental momentum states with a 3d basis vector."""


# Deprecated Alias, required to load files
FundamentalMomentumAxis3d = FundamentalTransformedPositionAxis3d
