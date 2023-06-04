from __future__ import annotations

import abc
from typing import Generic, Literal, Protocol, TypeVar

import numpy as np

_ND0Inv = TypeVar("_ND0Inv", bound=int)

AxisVector = np.ndarray[tuple[_ND0Inv], np.dtype[np.float_]]
AxisVector1d = AxisVector[Literal[3]]
AxisVector2d = AxisVector[Literal[3]]
AxisVector3d = AxisVector[Literal[3]]

_N0Inv = TypeVar("_N0Inv", bound=int)
_NF0Inv = TypeVar("_NF0Inv", bound=int)

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

    @property
    @abc.abstractmethod
    def vectors(self) -> np.ndarray[tuple[_N0Inv, _NF0Inv], np.dtype[np.complex_]]:
        ...


AxisLike1d = AxisLike[_NF0Inv, _N0Inv, Literal[1]]
AxisLike2d = AxisLike[_NF0Inv, _N0Inv, Literal[2]]
AxisLike3d = AxisLike[_NF0Inv, _N0Inv, Literal[3]]
