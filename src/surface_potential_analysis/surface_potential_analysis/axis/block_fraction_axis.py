# ruff: noqa: D102
from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Protocol, TypeVar

from surface_potential_analysis.axis.axis import FundamentalAxis

from .axis_like import AxisLike

if TYPE_CHECKING:
    import numpy as np

_N0_co = TypeVar("_N0_co", bound=int, covariant=True)
_NF0_co = TypeVar("_NF0_co", bound=int, covariant=True)


class AxisWithBlockFractionLike(AxisLike[_NF0_co, _N0_co], Protocol[_NF0_co, _N0_co]):  # type: ignore[misc]
    """A generic object that represents an axis with a corresponding axis vector."""

    @property
    @abc.abstractmethod
    def bloch_fractions(self) -> np.ndarray[tuple[_NF0_co, int], np.dtype[np.float_]]:
        ...


class ExplicitBlockFractionAxis(
    FundamentalAxis[_NF0_co], AxisWithBlockFractionLike[_NF0_co, _NF0_co]
):
    """A axis with vectors that are the fundamental position states."""

    def __init__(
        self, bloch_fractions: np.ndarray[tuple[_NF0_co, int], np.dtype[np.float_]]
    ) -> None:
        self._bloch_fractions = bloch_fractions
        super().__init__(bloch_fractions.size)  # type: ignore[arg-type]

    @property
    def bloch_fractions(self) -> np.ndarray[tuple[_NF0_co, int], np.dtype[np.float_]]:
        return self._bloch_fractions
