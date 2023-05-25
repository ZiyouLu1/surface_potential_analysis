from __future__ import annotations

from typing import TypeVar

import numpy as np

from surface_potential_analysis.basis.basis_like import BasisLike, BasisVector

_N0Inv = TypeVar("_N0Inv", bound=int)

_NF0Inv = TypeVar("_NF0Inv", bound=int)

# ruff: noqa: D102


class ExplicitBasis(BasisLike[_NF0Inv, _N0Inv]):
    """A basis which stores it's vectors explicitly as states in position space."""

    def __init__(
        self,
        delta_x: BasisVector,
        vectors: np.ndarray[tuple[_N0Inv, _NF0Inv], np.dtype[np.complex_]],
    ) -> None:
        self._delta_x = delta_x
        self._vectors = vectors
        super().__init__()

    @property
    def delta_x(self) -> BasisVector:
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


class FundamentalPositionBasis(BasisLike[_NF0Inv, _NF0Inv]):
    """A basis whos eigenstates are the fundamental position states."""

    def __init__(self, delta_x: BasisVector, n: _NF0Inv) -> None:
        self._delta_x = delta_x
        self._n = n
        super().__init__()

    @property
    def delta_x(self) -> BasisVector:
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


class MomentumBasis(BasisLike[_NF0Inv, _N0Inv]):
    """A basis with the n lowest frequency momentum states."""

    def __init__(self, delta_x: BasisVector, n: _N0Inv, fundamental_n: _NF0Inv) -> None:
        self._delta_x = delta_x
        self._n = n
        self._fundamental_n = fundamental_n
        super().__init__()

    @property
    def delta_x(self) -> BasisVector:
        return self._delta_x

    @property
    def n(self) -> _N0Inv:
        return self._n

    @property
    def fundamental_n(self) -> _NF0Inv:
        return self._fundamental_n

    @property
    def vectors(self) -> np.ndarray[tuple[_N0Inv, _NF0Inv], np.dtype[np.complex_]]:
        return np.fft.ifft(  # type: ignore[no-any-return]
            np.eye(self.n, self._fundamental_n), axis=1, norm="ortho"
        )


class FundamentalMomentumBasis(MomentumBasis[_NF0Inv, _NF0Inv]):
    """A basis who's eigenstates are the fundamental momentum states."""

    def __init__(self, delta_x: BasisVector, n: _NF0Inv) -> None:
        super().__init__(delta_x, n, n)
