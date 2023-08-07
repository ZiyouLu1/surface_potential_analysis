from __future__ import annotations

import abc
from typing import Literal, Protocol, TypeVar, runtime_checkable

import numpy as np

_N0Cov = TypeVar("_N0Cov", bound=int, covariant=True)
_NF0Cov = TypeVar("_NF0Cov", bound=int, covariant=True)

_ND0Inv = TypeVar("_ND0Inv", bound=int)
_N0Inv = TypeVar("_N0Inv", bound=int)
_NF0Inv = TypeVar("_NF0Inv", bound=int)

AxisVector = np.ndarray[tuple[_ND0Inv], np.dtype[np.float_]]
AxisVector1d = AxisVector[Literal[1]]
AxisVector2d = AxisVector[Literal[2]]
AxisVector3d = AxisVector[Literal[3]]


_S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])


# ruff: noqa: D102
@runtime_checkable
class FromFundamentalAxis(Protocol[_NF0Cov, _N0Cov]):
    """Represents an axis which can be converted from the fundamental axis."""

    @abc.abstractmethod
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
        ...


@runtime_checkable
class FromTransformedAxis(Protocol[_NF0Cov, _N0Cov]):
    """Represents an axis which can be converted from the transformed axis."""

    @abc.abstractmethod
    def __from_transformed__(
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
        ...


@runtime_checkable
class IntoFundamentalAxis(Protocol[_NF0Cov, _N0Cov]):
    """Represents an axis which can be converted to fundamental axis."""

    @abc.abstractmethod
    def __into_fundamental__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_ | np.float_]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
        ...


@runtime_checkable
class IntoTransformedAxis(Protocol[_NF0Cov, _N0Cov]):
    """Represents an axis which can be converted to transformed axis."""

    @abc.abstractmethod
    def __into_transformed__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_ | np.float_]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
        ...


@runtime_checkable
class AsFundamentalAxis(
    IntoFundamentalAxis[_NF0Cov, _N0Cov],
    IntoTransformedAxis[_NF0Cov, _N0Cov],
    Protocol[_NF0Cov, _N0Cov],
):
    """Represents an axis which can (inexpensively) be converted to fundamental axis."""

    @abc.abstractmethod
    def __as_fundamental__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_ | np.float_]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
        ...

    def __into_fundamental__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_ | np.float_]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
        return self.__as_fundamental__(vectors, axis)

    def __into_transformed__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_ | np.float_]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
        fundamental = self.__into_fundamental__(vectors, axis)
        return np.fft.fft(fundamental, axis=axis, norm="ortho")  # type: ignore[no-any-return]


@runtime_checkable
class AsTransformedAxis(
    IntoFundamentalAxis[_NF0Cov, _N0Cov],
    IntoTransformedAxis[_NF0Cov, _N0Cov],
    Protocol[_NF0Cov, _N0Cov],
):
    """Represents an axis which can (inexpensively) be converted to transformed axis."""

    @abc.abstractmethod
    def __as_transformed__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_ | np.float_]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
        ...

    def __into_transformed__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_ | np.float_]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
        return self.__as_transformed__(vectors, axis)

    def __into_fundamental__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_ | np.float_]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
        as_transformed = self.__into_transformed__(vectors, axis)
        return np.fft.ifft(as_transformed, axis=axis, norm="ortho")  # type: ignore[no-any-return]


class AxisLike(
    FromFundamentalAxis[_NF0Cov, _N0Cov],
    IntoFundamentalAxis[_NF0Cov, _N0Cov],
    FromTransformedAxis[_NF0Cov, _N0Cov],
    IntoTransformedAxis[_NF0Cov, _N0Cov],
    Protocol[_NF0Cov, _N0Cov],
):
    """A generic object that represents an axis for a axis."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n={self.n.__repr__()}, fundamental_n={self.fundamental_n.__repr__()})"

    @property
    @abc.abstractmethod
    def n(self) -> _N0Cov:
        ...

    @property
    @abc.abstractmethod
    def fundamental_n(self) -> _NF0Cov:
        ...

    def __into_fundamental__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_ | np.float_]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
        transformed = self.__into_transformed__(vectors, axis)
        return np.fft.ifft(transformed, axis=axis, norm="ortho")  # type: ignore[no-any-return]

    # !Can also be done like
    # !basis_vectors = self.__into_fundamental__(np.eye(self.n, self.n))
    # !transformed = np.tensordot(vectors, basis_vectors, axes=([axis], [1]))
    # !return np.moveaxis(transformed, -1, axis)  # type: ignore[no-any-return]
    def __into_transformed__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_ | np.float_]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
        fundamental = self.__into_fundamental__(vectors, axis)
        return np.fft.fft(fundamental, axis=axis, norm="ortho")  # type: ignore[no-any-return]

    def __from_fundamental__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_ | np.float_]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
        transformed = np.fft.fft(vectors, self.fundamental_n, axis=axis, norm="ortho")
        return self.__from_transformed__(transformed, axis)

    def __from_transformed__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_ | np.float_]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
        fundamental = np.fft.ifft(vectors, self.fundamental_n, axis=axis, norm="ortho")
        return self.__from_fundamental__(fundamental, axis)


class AxisWithLengthLike(AxisLike[_NF0Cov, _N0Cov], Protocol[_NF0Cov, _N0Cov, _ND0Inv]):
    """A generic object that represents an axis with a corresponding axis vector."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(delta_x={self.delta_x.__repr__()}, n={self.n.__repr__()}, fundamental_n={self.fundamental_n.__repr__()})"

    @property
    @abc.abstractmethod
    def delta_x(self) -> AxisVector[_ND0Inv]:
        ...


AxisWithLengthLike1d = AxisWithLengthLike[_NF0Inv, _N0Inv, Literal[1]]
AxisWithLengthLike2d = AxisWithLengthLike[_NF0Inv, _N0Inv, Literal[2]]
AxisWithLengthLike3d = AxisWithLengthLike[_NF0Inv, _N0Inv, Literal[3]]
