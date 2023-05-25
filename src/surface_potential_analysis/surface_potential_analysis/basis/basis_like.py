from __future__ import annotations

import abc
from typing import Generic, Literal, Protocol, TypeVar

import numpy as np

BasisVector = np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]

_N0Cov = TypeVar("_N0Cov", bound=int)
_NF0Inv = TypeVar("_NF0Inv", bound=int)


# ruff: noqa D102


class BasisLike(Protocol, Generic[_NF0Inv, _N0Cov]):
    """A generic object that represents a basis."""

    @property
    @abc.abstractmethod
    def delta_x(self) -> BasisVector:
        ...

    @property
    @abc.abstractmethod
    def n(self) -> _N0Cov:
        ...

    @property
    @abc.abstractmethod
    def fundamental_n(self) -> _NF0Inv:
        ...

    @property
    @abc.abstractmethod
    def vectors(self) -> np.ndarray[tuple[_N0Cov, _NF0Inv], np.dtype[np.complex_]]:
        ...
