from __future__ import annotations

from typing import Literal, TypeVar

import numpy as np

from surface_potential_analysis.axis.axis_like import (
    AsFundamentalBasis,
    AsTransformedBasis,
    AxisVector,
    BasisLike,
    BasisWithLengthLike,
)
from surface_potential_analysis.util.interpolation import pad_ft_points

_NF0_co = TypeVar("_NF0_co", bound=int, covariant=True)
_N0_co = TypeVar("_N0_co", bound=int, covariant=True)

_NF0Inv = TypeVar("_NF0Inv", bound=int)
_N0Inv = TypeVar("_N0Inv", bound=int)
_ND0Inv = TypeVar("_ND0Inv", bound=int)

_S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])
# ruff: noqa: D102


class ExplicitBasis(BasisWithLengthLike[_NF0_co, _N0_co, _ND0Inv]):
    """An basis with vectors given as explicit states."""

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
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_] | np.dtype[np.float_]],
        basis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
        transformed = np.tensordot(vectors, self.vectors, axes=([basis], [0]))
        return np.moveaxis(transformed, -1, basis)  # type: ignore[no-any-return]

    @classmethod
    def from_momentum_vectors(
        cls: type[ExplicitBasis[_NF0_co, _N0_co, _ND0Inv]],
        delta_x: AxisVector[_ND0Inv],
        vectors: np.ndarray[tuple[_N0_co, _NF0_co], np.dtype[np.complex_]],
    ) -> ExplicitBasis[_NF0_co, _N0_co, _ND0Inv]:
        vectors = np.fft.ifft(vectors, axis=1, norm="ortho")
        return cls(delta_x, vectors)


ExplicitBasis1d = ExplicitBasis[_NF0Inv, _N0Inv, Literal[1]]
"""An basis with vectors given as explicit states with a 1d basis vector."""

ExplicitBasis2d = ExplicitBasis[_NF0Inv, _N0Inv, Literal[2]]
"""An basis with vectors given as explicit states with a 2d basis vector."""

ExplicitBasis3d = ExplicitBasis[_NF0Inv, _N0Inv, Literal[3]]
"""An basis with vectors given as explicit states with a 3d basis vector."""


class FundamentalBasis(
    AsFundamentalBasis[_NF0_co, _NF0_co], BasisLike[_NF0_co, _NF0_co]
):
    """A basis with vectors that are the fundamental position states."""

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
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_] | np.dtype[np.float_]],
        basis: int = -1,
    ) -> np.ndarray[_S0Inv, np.dtype[np.complex_]]:
        return vectors.astype(np.complex_, copy=False)  # type: ignore[no-any-return]

    def __from_fundamental__(  # type: ignore[override]
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_] | np.dtype[np.float_]],
        basis: int = -1,
    ) -> np.ndarray[_S0Inv, np.dtype[np.complex_]]:
        return vectors.astype(np.complex_, copy=False)  # type: ignore[no-any-return]


class FundamentalPositionBasis(
    FundamentalBasis[_NF0_co], BasisWithLengthLike[_NF0_co, _NF0_co, _ND0Inv]
):
    """A basis with vectors that are the fundamental position states."""

    def __init__(self, delta_x: AxisVector[_ND0Inv], n: _NF0_co) -> None:
        self._delta_x = delta_x
        super().__init__(n)

    @property
    def delta_x(self) -> AxisVector[_ND0Inv]:
        return self._delta_x


FundamentalPositionBasis1d = FundamentalPositionBasis[_NF0Inv, Literal[1]]
"""A basis with vectors that are the fundamental position states with a 1d basis vector."""
FundamentalPositionBasis2d = FundamentalPositionBasis[_NF0Inv, Literal[2]]
"""A basis with vectors that are the fundamental position states with a 2d basis vector."""
FundamentalPositionBasis3d = FundamentalPositionBasis[_NF0Inv, Literal[3]]
"""A basis with vectors that are the fundamental position states with a 3d basis vector."""


class TransformedBasis(
    AsTransformedBasis[_NF0_co, _N0_co],
    BasisLike[_NF0_co, _N0_co],
):
    """A basis with vectors which are the n lowest frequency momentum states."""

    def __init__(self, n: _N0_co, fundamental_n: _NF0_co) -> None:
        self._n = n
        self._fundamental_n = fundamental_n
        assert self._fundamental_n >= self.n
        super().__init__()

    @property
    def n(self) -> _N0_co:
        return self._n

    @property
    def fundamental_n(self) -> _NF0_co:
        return self._fundamental_n

    def __as_transformed__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_] | np.dtype[np.float_]],
        basis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
        casted = vectors.astype(np.complex_, copy=False)
        return pad_ft_points(casted, s=(self.fundamental_n,), axes=(basis,))

    def __from_transformed__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_] | np.dtype[np.float_]],
        basis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
        casted = vectors.astype(np.complex_, copy=False)
        return pad_ft_points(casted, s=(self.n,), axes=(basis,))


class FundamentalTransformedBasis(TransformedBasis[_NF0_co, _NF0_co]):
    """A basis with vectors which are the n lowest frequency momentum states."""

    def __init__(self, n: _NF0_co) -> None:
        super().__init__(n, n)

    def __as_transformed__(  # type: ignore[override]
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_] | np.dtype[np.float_]],
        basis: int = -1,
    ) -> np.ndarray[_S0Inv, np.dtype[np.complex_]]:
        return vectors.astype(np.complex_, copy=False)  # type: ignore[no-any-return]

    def __from_transformed__(  # type: ignore[override]
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_] | np.dtype[np.float_]],
        basis: int = -1,
    ) -> np.ndarray[_S0Inv, np.dtype[np.complex_]]:
        return vectors.astype(np.complex_, copy=False)  # type: ignore[no-any-return]


class TransformedPositionBasis(
    TransformedBasis[_NF0_co, _N0_co],
    BasisWithLengthLike[_NF0_co, _N0_co, _ND0Inv],
):
    """A basis with vectors which are the n lowest frequency momentum states."""

    def __init__(
        self, delta_x: AxisVector[_ND0Inv], n: _N0_co, fundamental_n: _NF0_co
    ) -> None:
        self._delta_x = delta_x
        TransformedBasis.__init__(self, n, fundamental_n)  # type: ignore can't infer type

    @property
    def delta_x(self) -> AxisVector[_ND0Inv]:
        return self._delta_x


TransformedPositionBasis1d = TransformedPositionBasis[_NF0Inv, _N0Inv, Literal[1]]
"""A basis with vectors which are the n lowest frequency momentum states with a 1d basis vector."""


TransformedPositionBasis2d = TransformedPositionBasis[_NF0Inv, _N0Inv, Literal[2]]
"""A basis with vectors which are the n lowest frequency momentum states with a 2d basis vector."""


TransformedPositionBasis3d = TransformedPositionBasis[_NF0Inv, _N0Inv, Literal[3]]
"""A basis with vectors which are the n lowest frequency momentum states with a 3d basis vector."""


class FundamentalTransformedPositionBasis(
    TransformedPositionBasis[_NF0_co, _NF0_co, _ND0Inv],
    FundamentalTransformedBasis[_NF0_co],
):
    """An basis with vectors which are the fundamental momentum states."""

    def __init__(self, delta_x: AxisVector[_ND0Inv], n: _NF0_co) -> None:
        TransformedPositionBasis.__init__(self, delta_x, n, n)  # type: ignore can't infer type


FundamentalTransformedPositionBasis1d = FundamentalTransformedPositionBasis[
    _NF0Inv, Literal[1]
]
"""An basis with vectors which are the fundamental momentum states with a 1d basis vector."""


FundamentalTransformedPositionBasis2d = FundamentalTransformedPositionBasis[
    _NF0Inv, Literal[2]
]
"""An basis with vectors which are the fundamental momentum states with a 2d basis vector."""

FundamentalTransformedPositionBasis3d = FundamentalTransformedPositionBasis[
    _NF0Inv, Literal[3]
]
"""An basis with vectors which are the fundamental momentum states with a 3d basis vector."""
