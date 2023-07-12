from __future__ import annotations

from functools import cached_property
from typing import TypeVar

import numpy as np

from .axis_like import AxisLike, AxisVector, AxisWithLengthLike

_NF0Inv = TypeVar("_NF0Inv", bound=int)
_N0Inv = TypeVar("_N0Inv", bound=int)
_ND0Inv = TypeVar("_ND0Inv", bound=int)
_S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])


# ruff: noqa: D102
class AxisUtil(AxisLike[_NF0Inv, _N0Inv]):
    """A class to help with the manipulation of an axis."""

    _basis: AxisLike[_NF0Inv, _N0Inv]

    def __init__(self, basis: AxisLike[_NF0Inv, _N0Inv]) -> None:
        self._basis = basis

    @property
    def n(self) -> _N0Inv:
        return self._basis.n

    @property
    def fundamental_n(self) -> _NF0Inv:
        return self._basis.fundamental_n

    @property
    def vectors(self) -> np.ndarray[tuple[_N0Inv, _NF0Inv], np.dtype[np.complex_]]:
        return self.__into_fundamental__(np.eye(self.n, self.n))  # type: ignore[return-value]

    def __into_fundamental__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_ | np.float_]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
        return self._basis.__into_fundamental__(vectors, axis)

    def __from_fundamental__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_ | np.float_]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
        return self._basis.__from_fundamental__(vectors, axis)

    @property
    def nx_points(self) -> np.ndarray[tuple[_N0Inv], np.dtype[np.int_]]:
        return np.arange(0, self.n, dtype=int)  # type: ignore[no-any-return]

    @property
    def nk_points(self) -> np.ndarray[tuple[_N0Inv], np.dtype[np.int_]]:
        return np.fft.ifftshift(  # type: ignore[no-any-return]
            np.arange((-self.n + 1) // 2, (self.n + 1) // 2)
        )

    @property
    def fundamental_nk_points(self) -> np.ndarray[tuple[_NF0Inv], np.dtype[np.int_]]:
        # We want points from (-self.Nk + 1) // 2 to (self.Nk - 1) // 2
        n = self.fundamental_n
        return np.fft.ifftshift(  # type: ignore[no-any-return]
            np.arange((-n + 1) // 2, (n + 1) // 2)
        )

    @property
    def fundamental_nx_points(self) -> np.ndarray[tuple[_NF0Inv], np.dtype[np.int_]]:
        return np.arange(  # type: ignore[no-any-return]
            0, self.fundamental_n, dtype=int  # type: ignore[misc]
        )


class AxisWithLengthLikeUtil(
    AxisUtil[_NF0Inv, _N0Inv], AxisWithLengthLike[_NF0Inv, _N0Inv, _ND0Inv]
):
    """A class to help with the manipulation of an axis."""

    _basis: AxisWithLengthLike[_NF0Inv, _N0Inv, _ND0Inv]

    def __init__(self, basis: AxisWithLengthLike[_NF0Inv, _N0Inv, _ND0Inv]) -> None:
        super().__init__(basis)

    @property
    def delta_x(self) -> AxisVector[_ND0Inv]:
        return self._basis.delta_x

    @cached_property
    def dx(self) -> AxisVector[_ND0Inv]:
        return self.delta_x / self.n  # type: ignore[no-any-return, misc]

    @cached_property
    def fundamental_dx(self) -> AxisVector[_ND0Inv]:
        return self.delta_x / self.fundamental_n  # type: ignore[no-any-return,misc]

    @property
    def x_points(self) -> np.ndarray[tuple[_ND0Inv, _N0Inv], np.dtype[np.int_]]:
        return self.dx[:, np.newaxis] * self.nx_points  # type: ignore[no-any-return]

    @property
    def fundamental_x_points(
        self,
    ) -> np.ndarray[tuple[_ND0Inv, _NF0Inv], np.dtype[np.int_]]:
        return self.fundamental_dx[:, np.newaxis] * self.fundamental_nx_points  # type: ignore[no-any-return]
