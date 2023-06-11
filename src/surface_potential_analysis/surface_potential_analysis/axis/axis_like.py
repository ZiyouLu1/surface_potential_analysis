from __future__ import annotations

import abc
from typing import Generic, Literal, Protocol, TypeVar

import numpy as np

_ND0Inv = TypeVar("_ND0Inv", bound=int)
_N0Inv = TypeVar("_N0Inv", bound=int)
_NF0Inv = TypeVar("_NF0Inv", bound=int)

_ND0Cov = TypeVar("_ND0Cov", bound=int, covariant=True)
_N0Cov = TypeVar("_N0Cov", bound=int, covariant=True)
_NF0Cov = TypeVar("_NF0Cov", bound=int, covariant=True)


AxisVector = np.ndarray[tuple[_ND0Inv], np.dtype[np.float_]]
AxisVector1d = AxisVector[Literal[3]]
AxisVector2d = AxisVector[Literal[3]]
AxisVector3d = AxisVector[Literal[3]]


_S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])
# ruff: noqa: D102


class AxisLike(Protocol, Generic[_NF0Inv, _N0Inv, _ND0Inv]):
    """A generic object that represents an axis for a basis."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(delta_x={self.delta_x.__repr__()}, n={self.n.__repr__()}, fundamental_n={self.fundamental_n.__repr__()})"

    @property
    @abc.abstractmethod
    def delta_x(self) -> AxisVector[_ND0Inv]:
        ...

    @property
    @abc.abstractmethod
    def n(self) -> _N0Inv:
        ...

    @property
    @abc.abstractmethod
    def fundamental_n(self) -> _NF0Inv:
        ...

    @abc.abstractmethod
    def __into_fundamental__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_ | np.float_]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
        ...

    def __from_fundamental__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_ | np.float_]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
        """
        Given a set of vectors convert them to fundamental basis along the given axis.

        Parameters
        ----------
        vectors : np.ndarray[_S0Inv, np.dtype[np.complex_  |  np.float_]]
        axis : int, optional
            axis to transform, by default -1

        Returns
        -------
        np.ndarray[tuple[int, ...], np.dtype[np.complex_]]
            The vectors, converted along axis
        """
        basis_vectors = self.__into_fundamental__(np.eye(self.n, self.n))
        transformed = np.tensordot(vectors, basis_vectors, axes=([axis], [1]))
        return np.moveaxis(transformed, -1, axis)  # type: ignore[no-any-return]


AxisLike1d = AxisLike[_NF0Inv, _N0Inv, Literal[1]]
AxisLike2d = AxisLike[_NF0Inv, _N0Inv, Literal[2]]
AxisLike3d = AxisLike[_NF0Inv, _N0Inv, Literal[3]]
