from __future__ import annotations

import abc
from typing import Any, Literal, Protocol, TypeVar, runtime_checkable

import numpy as np

_N0_co = TypeVar("_N0_co", bound=int, covariant=True)
_NF0_co = TypeVar("_NF0_co", bound=int, covariant=True)

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
class FromFundamentalAxis(Protocol[_NF0_co, _N0_co]):
    """Represents an axis which can be converted from the fundamental axis."""

    @abc.abstractmethod
    def __from_fundamental__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_] | np.dtype[np.float_]],
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
class FromTransformedAxis(Protocol[_NF0_co, _N0_co]):
    """Represents an axis which can be converted from the transformed axis."""

    @abc.abstractmethod
    def __from_transformed__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_] | np.dtype[np.float_]],
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
class IntoFundamentalAxis(Protocol[_NF0_co, _N0_co]):
    """Represents an axis which can be converted to fundamental axis."""

    @abc.abstractmethod
    def __into_fundamental__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_] | np.dtype[np.float_]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
        ...


@runtime_checkable
class IntoTransformedAxis(Protocol[_NF0_co, _N0_co]):
    """Represents an axis which can be converted to transformed axis."""

    @abc.abstractmethod
    def __into_transformed__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_] | np.dtype[np.float_]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
        ...


@runtime_checkable
class AsFundamentalBasis(
    IntoFundamentalAxis[_NF0_co, _N0_co],
    IntoTransformedAxis[_NF0_co, _N0_co],
    Protocol[_NF0_co, _N0_co],
):
    """Represents an axis which can (inexpensively) be converted to fundamental axis."""

    @abc.abstractmethod
    def __as_fundamental__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_] | np.dtype[np.float_]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
        ...

    def __into_fundamental__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_] | np.dtype[np.float_]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
        return self.__as_fundamental__(vectors, axis)

    def __into_transformed__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_] | np.dtype[np.float_]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
        fundamental = self.__into_fundamental__(vectors, axis)
        return np.fft.fft(fundamental, axis=axis, norm="ortho")  # type: ignore[no-any-return]


@runtime_checkable
class AsTransformedBasis(
    IntoFundamentalAxis[_NF0_co, _N0_co],
    IntoTransformedAxis[_NF0_co, _N0_co],
    Protocol[_NF0_co, _N0_co],
):
    """Represents an axis which can (inexpensively) be converted to transformed axis."""

    @abc.abstractmethod
    def __as_transformed__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_] | np.dtype[np.float_]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
        ...

    def __into_transformed__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_] | np.dtype[np.float_]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
        return self.__as_transformed__(vectors, axis)

    def __into_fundamental__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_] | np.dtype[np.float_]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
        as_transformed = self.__into_transformed__(vectors, axis)
        return np.fft.ifft(as_transformed, axis=axis, norm="ortho")  # type: ignore[no-any-return]


@runtime_checkable
class BasisLike(
    FromFundamentalAxis[_NF0_co, _N0_co],
    IntoFundamentalAxis[_NF0_co, _N0_co],
    FromTransformedAxis[_NF0_co, _N0_co],
    IntoTransformedAxis[_NF0_co, _N0_co],
    Protocol[_NF0_co, _N0_co],
):
    """A generic object that represents an axis for a axis."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n={self.n.__repr__()}, fundamental_n={self.fundamental_n.__repr__()})"

    @property
    @abc.abstractmethod
    def n(self) -> _N0_co:
        ...

    @property
    @abc.abstractmethod
    def fundamental_n(self) -> _NF0_co:
        ...

    def __into_fundamental__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_] | np.dtype[np.float_]],
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
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_] | np.dtype[np.float_]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
        fundamental = self.__into_fundamental__(vectors, axis)
        return np.fft.fft(fundamental, axis=axis, norm="ortho")  # type: ignore[no-any-return]

    def __from_fundamental__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_] | np.dtype[np.float_]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
        transformed = np.fft.fft(vectors, self.fundamental_n, axis=axis, norm="ortho")
        return self.__from_transformed__(transformed, axis)

    def __from_transformed__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex_] | np.dtype[np.float_]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
        fundamental = np.fft.ifft(vectors, self.fundamental_n, axis=axis, norm="ortho")
        return self.__from_fundamental__(fundamental, axis)

    def __convert_vector_into__(
        self,
        vector: np.ndarray[_S0Inv, np.dtype[np.complex_] | np.dtype[np.float_]],
        basis: BasisLike[Any, Any],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[np.complex_]]:
        assert basis.fundamental_n == self.fundamental_n
        # Small speedup here, and prevents imprecision of fft followed by ifft
        # And two pad_ft_points
        if isinstance(self, AsTransformedBasis) and isinstance(
            basis, AsTransformedBasis
        ):
            # If initial axis and final axis are AsTransformedAxis
            # we (might) be able to prevent the need for a fft
            transformed = self.__into_transformed__(vector, axis)
            return basis.__from_transformed__(transformed, axis)
        fundamental = self.__into_fundamental__(vector, axis)
        return basis.__from_fundamental__(fundamental, axis)


def convert_vector(
    vector: np.ndarray[_S0Inv, np.dtype[np.complex_] | np.dtype[np.float_]],
    initial_basis: BasisLike[Any, Any],
    final_basis: BasisLike[Any, Any],
    axis: int = -1,
) -> np.ndarray[Any, np.dtype[np.complex_]]:
    """
    Convert a vector, expressed in terms of the given basis from_config in the basis to_config.

    Parameters
    ----------
    vector : np.ndarray[tuple[int], np.dtype[np.complex_] | np.dtype[np.float_]]
        the vector to convert
    from_config : _B3d0Inv
    to_config : _B3d1Inv
    axis : int, optional
        axis along which to convert, by default -1

    Returns
    -------
    np.ndarray[tuple[int], np.dtype[np.complex_]]
    """
    return initial_basis.__convert_vector_into__(vector, final_basis, axis)


def convert_dual_vector(
    co_vector: np.ndarray[_S0Inv, np.dtype[np.complex_] | np.dtype[np.float_]],
    initial_basis: BasisLike[Any, Any],
    final_basis: BasisLike[Any, Any],
    axis: int = -1,
) -> np.ndarray[tuple[int, ...], np.dtype[np.complex_]]:
    """
    Convert a co_vector, expressed in terms of the given basis from_config in the basis to_config.

    Parameters
    ----------
    co_vector : np.ndarray[tuple[int], np.dtype[np.complex_]]
        the vector to convert
    from_config : _B3d0Inv
    to_config : _B3d1Inv
    axis : int, optional
        axis along which to convert, by default -1

    Returns
    -------
    np.ndarray[tuple[int], np.dtype[np.complex_]]
    """
    return np.conj(convert_vector(np.conj(co_vector), initial_basis, final_basis, axis))  # type: ignore[no-any-return]


def convert_matrix(
    matrix: np.ndarray[tuple[int, int], np.dtype[np.complex_] | np.dtype[np.float_]],
    initial_basis: BasisLike[Any, Any],
    final_basis: BasisLike[Any, Any],
    initial_dual_basis: BasisLike[Any, Any],
    final_dual_basis: BasisLike[Any, Any],
) -> np.ndarray[tuple[int, int], np.dtype[np.complex_]]:
    """
    Convert a matrix from initial_basis to final_basis.

    Parameters
    ----------
    matrix : np.ndarray[tuple[int, int], np.dtype[np.complex_]]
    initial_basis : _B3d0Inv
    final_basis : _B3d1Inv

    Returns
    -------
    np.ndarray[tuple[int, int], np.dtype[np.complex_]]
    """
    converted = convert_vector(matrix, initial_basis, final_basis, axis=0)
    return convert_dual_vector(converted, initial_dual_basis, final_dual_basis, axis=1)  # type: ignore[return-value]


@runtime_checkable
class BasisWithLengthLike(
    BasisLike[_NF0_co, _N0_co], Protocol[_NF0_co, _N0_co, _ND0Inv]
):
    """A generic object that represents an axis with a corresponding axis vector."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(delta_x={self.delta_x.__repr__()}, n={self.n.__repr__()}, fundamental_n={self.fundamental_n.__repr__()})"

    @property
    @abc.abstractmethod
    def delta_x(self) -> AxisVector[_ND0Inv]:
        ...


AxisWithLengthLike1d = BasisWithLengthLike[_NF0Inv, _N0Inv, Literal[1]]
AxisWithLengthLike2d = BasisWithLengthLike[_NF0Inv, _N0Inv, Literal[2]]
AxisWithLengthLike3d = BasisWithLengthLike[_NF0Inv, _N0Inv, Literal[3]]
