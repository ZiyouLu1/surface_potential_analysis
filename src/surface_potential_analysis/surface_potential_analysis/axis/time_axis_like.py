from __future__ import annotations

import abc
from typing import Protocol, TypeVar

import numpy as np

from surface_potential_analysis.axis.axis import EvenlySpacedAxis, FundamentalAxis
from surface_potential_analysis.axis.axis_like import AxisLike

_N0_co = TypeVar("_N0_co", bound=int, covariant=True)
_NF0_co = TypeVar("_NF0_co", bound=int, covariant=True)


# ruff: noqa: D102
class AxisWithTimeLike(AxisLike[_NF0_co, _N0_co], Protocol[_NF0_co, _N0_co]):  # type: ignore[misc]
    """A generic object that represents an axis with a corresponding axis vector."""

    @property
    @abc.abstractmethod
    def times(self) -> np.ndarray[tuple[_N0_co], np.dtype[np.float_]]:
        ...

    @property
    @abc.abstractmethod
    def fundamental_times(self) -> np.ndarray[tuple[_NF0_co], np.dtype[np.float_]]:
        ...

    @property
    def delta_t(self) -> float:
        return self.fundamental_times[1] - self.fundamental_times[0]  # type: ignore[no-any-return]


class EvenlySpacedTimeAxis(
    EvenlySpacedAxis[_NF0_co, _N0_co], AxisWithTimeLike[_NF0_co, _N0_co]
):
    """A axis with vectors that are the fundamental position states."""

    def __init__(self, n: _NF0_co, step: int, delta_t: float) -> None:
        self._delta_t = delta_t
        super().__init__(n, step)  # type: ignore[arg-type]

    @property
    def times(self) -> np.ndarray[tuple[_N0_co], np.dtype[np.float_]]:
        return np.linspace(0, self._delta_t, self.n)  # type: ignore[no-any-return]

    @property
    def fundamental_times(self) -> np.ndarray[tuple[_NF0_co], np.dtype[np.float_]]:
        return np.linspace(0, self._delta_t, self.fundamental_n)  # type: ignore[no-any-return]

    @property
    def delta_t(self) -> float:
        return self._delta_t


class FundamentalTimeAxis(EvenlySpacedTimeAxis[_NF0_co, _NF0_co]):
    """A axis with vectors that are the fundamental position states."""

    def __init__(self, n: _NF0_co, delta_t: float) -> None:
        self._delta_t = delta_t
        super().__init__(n, 1, delta_t)  # type: ignore[arg-type]


class ExplicitTimeAxis(FundamentalAxis[_NF0_co], AxisWithTimeLike[_NF0_co, _NF0_co]):
    """A axis with vectors that are the fundamental position states."""

    def __init__(self, times: np.ndarray[tuple[_NF0_co], np.dtype[np.float_]]) -> None:
        self._times = times
        super().__init__(times.size)  # type: ignore[arg-type]

    @property
    def times(self) -> np.ndarray[tuple[_NF0_co], np.dtype[np.float_]]:
        return self._times

    @property
    def fundamental_times(self) -> np.ndarray[tuple[_NF0_co], np.dtype[np.float_]]:
        return self.times
