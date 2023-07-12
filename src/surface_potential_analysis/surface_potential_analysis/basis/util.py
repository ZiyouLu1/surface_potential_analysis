from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, Unpack, overload

import numpy as np

from surface_potential_analysis.axis.axis import (
    FundamentalPositionAxis,
    FundamentalPositionAxis3d,
)
from surface_potential_analysis.axis.util import AxisUtil, AxisWithLengthLikeUtil
from surface_potential_analysis.util.util import (
    get_position_in_sorted,
    slice_ignoring_axes,
)

from .basis import AxisWithLengthBasis, Basis, Basis3d

if TYPE_CHECKING:
    from surface_potential_analysis._types import (
        ArrayFlatIndexLike,
        ArrayIndexLike,
        ArrayStackedIndexLike,
        FlatIndexLike,
        SingleFlatIndexLike,
        SingleIndexLike,
        SingleStackedIndexLike,
        StackedIndexLike,
        _IntLike_co,
    )



    _S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])
_B0Inv = TypeVar("_B0Inv", bound=Basis)
_ALB0Inv = TypeVar("_ALB0Inv", bound=AxisWithLengthBasis[Any])
_B3d0Inv = TypeVar("_B3d0Inv", bound=Basis3d[Any, Any, Any])
_NDInv = TypeVar("_NDInv", bound=int)


class BasisUtil(Generic[_B0Inv]):
    """
    A class to help with the manipulation of basis states.

    Note: The dimension of the axes must match the number of axes
    """

    _basis: _B0Inv

    def __init__(self, basis: _B0Inv) -> None:
        self._basis = basis

    @cached_property
    def _utils(self) -> tuple[AxisUtil[Any, Any], ...]:
        return tuple(AxisUtil(b) for b in self._basis)

    @property
    def nk_points(self) -> ArrayStackedIndexLike[tuple[int]]:
        nk_mesh = np.meshgrid(
            *[xi_axis.nk_points for xi_axis in self._utils],
            indexing="ij",
        )
        return tuple(nki.ravel() for nki in nk_mesh)

    @property
    def fundamental_nk_points(self) -> ArrayStackedIndexLike[tuple[int]]:
        nk_mesh = np.meshgrid(
            *[xi_axis.fundamental_nk_points for xi_axis in self._utils],
            indexing="ij",
        )
        return tuple(nki.ravel() for nki in nk_mesh)

    @property
    def nx_points(self) -> ArrayStackedIndexLike[tuple[int]]:
        nx_mesh = np.meshgrid(
            *[xi_axis.nx_points for xi_axis in self._utils],
            indexing="ij",
        )
        return tuple(nxi.ravel() for nxi in nx_mesh)

    @property
    def fundamental_nx_points(self) -> ArrayStackedIndexLike[tuple[int]]:
        nx_mesh = np.meshgrid(
            *[xi_axis.fundamental_nx_points for xi_axis in self._utils],
            indexing="ij",
        )
        return tuple(nxi.ravel() for nxi in nx_mesh)

    @cached_property
    def shape(self) -> tuple[int, ...]:
        return tuple(axi.n for axi in self._basis)

    @property
    def size(self) -> int:
        return np.prod(self.shape)  # type: ignore[return-value]

    @cached_property
    def fundamental_shape(self) -> tuple[int, ...]:
        return tuple(axi.fundamental_n for axi in self._basis)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @overload
    def get_flat_index(
        self,
        idx: SingleStackedIndexLike,
        *,
        mode: Literal["raise", "wrap", "clip"] = "raise",
    ) -> np.int_:
        ...

    @overload
    def get_flat_index(
        self,
        idx: ArrayStackedIndexLike[_S0Inv],
        *,
        mode: Literal["raise", "wrap", "clip"] = "raise",
    ) -> ArrayFlatIndexLike[_S0Inv]:
        ...

    def get_flat_index(
        self,
        idx: StackedIndexLike,
        *,
        mode: Literal["raise", "wrap", "clip"] = "raise",
    ) -> np.int_ | ArrayFlatIndexLike[_S0Inv]:
        """
        Given a stacked index, get the flat index into the Wigner-Seitz cell.

        Parameters
        ----------
        idx : tuple[int, int, int]
            The stacked index
        mode : Literal[&quot;raise&quot;, &quot;wrap&quot;, &quot;clip&quot;], optional
            Specifies how out-of-bounds indices are handled, by default "raise"

        Returns
        -------
        int
            the flattened index into the Wigner-Seitz cell
        """
        return np.ravel_multi_index(idx, self.shape, mode=mode)

    @overload
    def get_stacked_index(self, idx: SingleFlatIndexLike) -> SingleStackedIndexLike:
        ...

    @overload
    def get_stacked_index(
        self, idx: ArrayFlatIndexLike[_S0Inv]
    ) -> ArrayStackedIndexLike[_S0Inv]:
        ...

    def get_stacked_index(self, idx: FlatIndexLike) -> StackedIndexLike:
        """
        Given a flat index, produce a stacked index.

        Parameters
        ----------
        idx : int

        Returns
        -------
        tuple[int, int, int]
        """
        return np.unravel_index(idx, self.shape)  # type: ignore[return-value]


# ruff: noqa: D102
class AxisWithLengthBasisUtil(BasisUtil[_ALB0Inv]):
    """
    A class to help with the manipulation of basis states.

    Note: The dimension of the axes must match the number of axes
    """

    def __init__(self, basis: _ALB0Inv) -> None:
        if any(x.delta_x.size != len(basis) for x in basis):
            msg = "Basis has incorrect shape"
            raise AssertionError(msg)
        super().__init__(basis)

    @cached_property
    def _utils(self) -> tuple[AxisWithLengthLikeUtil[Any, Any, Any], ...]:
        return tuple(AxisWithLengthLikeUtil(b) for b in self._basis)

    @cached_property
    def volume(self) -> np.float_:
        return np.linalg.det(self.delta_x)  # type: ignore[no-any-return]

    @cached_property
    def reciprocal_volume(self) -> np.float_:
        return np.linalg.det(self.dk)  # type: ignore[no-any-return]

    @overload
    def get_k_points_at_index(
        self, idx: SingleIndexLike
    ) -> np.ndarray[tuple[int], np.dtype[np.float_]]:
        ...

    @overload
    def get_k_points_at_index(
        self, idx: ArrayIndexLike[_S0Inv]
    ) -> np.ndarray[tuple[int, Unpack[_S0Inv]], np.dtype[np.float_]]:
        ...

    def get_k_points_at_index(
        self, idx: ArrayIndexLike[_S0Inv] | SingleIndexLike
    ) -> (
        np.ndarray[tuple[int, Unpack[_S0Inv]], np.dtype[np.float_]]
        | np.ndarray[tuple[int], np.dtype[np.float_]]
    ):
        nk_points = idx if isinstance(idx, tuple) else self.get_stacked_index(idx)
        return np.tensordot(self.dk, nk_points, axes=(0, 0))  # type: ignore[no-any-return]

    @property
    def k_points(
        self,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float_]]:
        return self.get_k_points_at_index(self.nk_points)

    @property
    def fundamental_k_points(
        self,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float_]]:
        return np.tensordot(self.fundamental_dk, self.fundamental_nk_points, axes=(0, 0))  # type: ignore[no-any-return]

    @overload
    def get_x_points_at_index(
        self, idx: SingleIndexLike
    ) -> np.ndarray[tuple[int], np.dtype[np.float_]]:
        ...

    @overload
    def get_x_points_at_index(
        self, idx: ArrayIndexLike[_S0Inv]
    ) -> np.ndarray[tuple[int, Unpack[_S0Inv]], np.dtype[np.float_]]:
        ...

    def get_x_points_at_index(
        self, idx: ArrayIndexLike[_S0Inv] | SingleIndexLike
    ) -> (
        np.ndarray[tuple[int, Unpack[_S0Inv]], np.dtype[np.float_]]
        | np.ndarray[tuple[int], np.dtype[np.float_]]
    ):
        nx_points = idx if isinstance(idx, tuple) else self.get_stacked_index(idx)
        return np.tensordot(self.dx, nx_points, axes=(0, 0))  # type: ignore[no-any-return]

    @property
    def x_points(
        self,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float_]]:
        return self.get_x_points_at_index(self.nx_points)

    @property
    def fundamental_x_points(
        self,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float_]]:
        return np.tensordot(self.fundamental_dx, self.fundamental_nx_points, axes=(0, 0))  # type: ignore[no-any-return]

    @property
    def delta_x(self) -> np.ndarray[tuple[int, int], np.dtype[np.float_]]:
        return np.array([axi.delta_x for axi in self._basis])  # type: ignore[no-any-return]

    @property
    def fundamental_delta_x(self) -> np.ndarray[tuple[int, int], np.dtype[np.float_]]:
        return np.array([axi.delta_x for axi in self._basis])  # type: ignore[no-any-return]

    @cached_property
    def dx(self) -> np.ndarray[tuple[int, int], np.dtype[np.float_]]:
        return np.array([axi.dx for axi in self._utils])  # type: ignore[no-any-return]

    @cached_property
    def fundamental_dx(self) -> np.ndarray[tuple[int, int], np.dtype[np.float_]]:
        return np.array([axi.fundamental_dx for axi in self._utils])  # type: ignore[no-any-return]

    @property
    def delta_k(self) -> np.ndarray[tuple[int, int], np.dtype[np.float_]]:
        return np.array(self.shape)[:, np.newaxis] * self.dk  # type: ignore[no-any-return]

    @cached_property
    def fundamental_delta_k(self) -> np.ndarray[tuple[int, int], np.dtype[np.float_]]:
        return np.array(self.fundamental_shape)[:, np.newaxis] * self.dk  # type: ignore[no-any-return]

    @cached_property
    def dk(self) -> np.ndarray[tuple[int, int], np.dtype[np.float_]]:
        """Get dk as a list of dk for each axis."""
        return 2 * np.pi * np.linalg.inv(self.delta_x).T  # type: ignore[no-any-return]

    @property
    def fundamental_dk(self) -> np.ndarray[tuple[int, int], np.dtype[np.float_]]:
        return self.dk


def project_k_points_along_axes(
    points: np.ndarray[tuple[_NDInv, Unpack[_S0Inv]], np.dtype[np.float_]],
    basis: _ALB0Inv,
    axes: tuple[int, int],
) -> np.ndarray[tuple[Literal[2], Unpack[_S0Inv]], np.dtype[np.float_]]:
    """
    Get the list of k points projected onto the plane including both axes.

    Parameters
    ----------
    points : np.ndarray[tuple[int, Unpack[_S0Inv]], np.dtype[np.float_]]
    basis : _B0Inv
    axes : tuple[int, int]

    Returns
    -------
    np.ndarray[tuple[Literal[2], Unpack[_S0Inv]], np.dtype[np.float_]]
    """
    util = AxisWithLengthBasisUtil(basis)

    ax_0 = util.delta_k[axes[0]] / np.linalg.norm(util.delta_k[axes[0]])
    # Subtract off parallel componet
    ax_1 = util.delta_k[axes[1]] - np.tensordot(ax_0, util.delta_k[axes[1]], 1)
    ax_1 /= np.linalg.norm(ax_1)

    projected_0 = np.tensordot(ax_0, points, axes=(0, 0))
    projected_1 = np.tensordot(ax_1, points, axes=(0, 0))

    return np.array([projected_0, projected_1])  # type: ignore[no-any-return]


def get_fundamental_k_points_projected_along_axes(
    basis: _ALB0Inv,
    axes: tuple[int, int],
) -> np.ndarray[tuple[Literal[2], int], np.dtype[np.float_]]:
    """
    Get the fundamental_k_points projected onto the plane including both axes.

    Parameters
    ----------
    basis : _B0Inv
    axes : tuple[int, int]

    Returns
    -------
    np.ndarray[tuple[Literal[2], int], np.dtype[np.float_]]
    """
    util = AxisWithLengthBasisUtil(basis)
    points = util.fundamental_k_points
    return project_k_points_along_axes(points, basis, axes)


def get_k_coordinates_in_axes(
    basis: _ALB0Inv,
    axes: tuple[int, ...],
    idx: SingleStackedIndexLike | None,
) -> np.ndarray[tuple[Literal[2], int], np.dtype[np.float_]]:
    """
    Get the fundamental_k_points projected onto the plane including both axes.

    Parameters
    ----------
    basis : _B0Inv
    axes : tuple[int, int]

    Returns
    -------
    np.ndarray[tuple[Literal[2], int], np.dtype[np.float_]]
    """
    util = AxisWithLengthBasisUtil(basis)
    idx = tuple(0 for _ in range(util.ndim - len(axes))) if idx is None else idx
    points = get_fundamental_k_points_projected_along_axes(basis, axes[:2])  # type: ignore[arg-type]
    _slice = slice_ignoring_axes(idx, axes)
    return np.transpose(points.reshape(2, *util.shape)[:, *_slice], (0, *(1 + np.array(get_position_in_sorted(axes)))))  # type: ignore[no-any-return]


def project_x_points_along_axes(
    points: np.ndarray[tuple[_NDInv, Unpack[_S0Inv]], np.dtype[np.float_]],
    basis: _ALB0Inv,
    axes: tuple[int, int],
) -> np.ndarray[tuple[Literal[2], Unpack[_S0Inv]], np.dtype[np.float_]]:
    """
    Get the list of x points projected onto the plane including both axes.

    Parameters
    ----------
    points : np.ndarray[tuple[int, Unpack[_S0Inv]], np.dtype[np.float_]]
    basis : _B0Inv
    axes : tuple[int, int]

    Returns
    -------
    np.ndarray[tuple[Literal[2], Unpack[_S0Inv]], np.dtype[np.float_]]
    """
    util = AxisWithLengthBasisUtil(basis)

    ax_0 = util.delta_x[axes[0]] / np.linalg.norm(util.delta_x[axes[0]])
    # Subtract off parallel componet
    ax_1 = util.delta_x[axes[1]] - (ax_0 * np.dot(ax_0, util.delta_x[axes[1]]))
    ax_1 /= np.linalg.norm(ax_1)

    projected_0 = np.tensordot(ax_0, points, axes=(0, 0))
    projected_1 = np.tensordot(ax_1, points, axes=(0, 0))

    return np.array([projected_0, projected_1])  # type: ignore[no-any-return]


def get_fundamental_x_points_projected_along_axes(
    basis: _ALB0Inv,
    axes: tuple[int, int],
) -> np.ndarray[tuple[Literal[2], int], np.dtype[np.float_]]:
    """
    Get the fundamental_x_points projected onto the plane including both axes.

    Parameters
    ----------
    basis : _B0Inv
    axes : tuple[int, int]

    Returns
    -------
    np.ndarray[tuple[Literal[2], int], np.dtype[np.float_]]
    """
    util = AxisWithLengthBasisUtil(basis)
    points = util.fundamental_x_points
    return project_x_points_along_axes(points, basis, axes)


def get_x_coordinates_in_axes(
    basis: _ALB0Inv,
    axes: _S0Inv,
    idx: SingleStackedIndexLike | None,
) -> np.ndarray[tuple[Literal[2], int], np.dtype[np.float_]]:
    """
    Get the fundamental_x_points projected onto the plane including both axes.

    Parameters
    ----------
    basis : _B0Inv
    axes : tuple[int, int]

    Returns
    -------
    np.ndarray[tuple[Literal[2], int], np.dtype[np.float_]]
    """
    util = AxisWithLengthBasisUtil(basis)
    idx = tuple(0 for _ in range(util.ndim - len(axes))) if idx is None else idx
    points = get_fundamental_x_points_projected_along_axes(basis, axes[:2])  # type: ignore[arg-type]
    _slice = slice_ignoring_axes(idx, axes)
    return np.transpose(points.reshape(2, *util.shape)[:, *_slice], (0, *(1 + np.array(get_position_in_sorted(axes)))))  # type: ignore[no-any-return]


@overload
def _wrap_distance(distance: _IntLike_co, length: _IntLike_co) -> int:
    ...


@overload
def _wrap_distance(
    distance: np.ndarray[_S0Inv, np.dtype[np.int_]], length: _IntLike_co
) -> np.ndarray[_S0Inv, np.dtype[np.int_]]:
    ...


def _wrap_distance(distance: Any, length: _IntLike_co) -> Any:
    return np.subtract(np.mod(np.add(distance, length // 2), length), length // 2)


@overload
def wrap_index_around_origin(
    basis: _ALB0Inv,
    idx: SingleStackedIndexLike,
    origin: SingleIndexLike | None = None,
    axes: _S0Inv | None = None,
) -> SingleStackedIndexLike:
    ...


@overload
def wrap_index_around_origin(
    basis: _ALB0Inv,
    idx: ArrayStackedIndexLike[_S0Inv],
    origin: SingleIndexLike | None = None,
    axes: _S0Inv | None = None,
) -> ArrayStackedIndexLike[_S0Inv]:
    ...


def wrap_index_around_origin(
    basis: _ALB0Inv,
    idx: StackedIndexLike,
    origin: SingleIndexLike | None = None,
    axes: _S0Inv | None = None,
) -> StackedIndexLike:
    """
    Given an index or list of indexes in stacked form, find the equivalent index closest to the origin.

    Parameters
    ----------
    basis : _B3d0Inv
    idx : StackedIndexLike | FlatIndexLike
    origin_idx : StackedIndexLike | FlatIndexLike, optional
        origin to wrap around, by default (0, 0, 0)

    Returns
    -------
    StackedIndexLike
    """
    util = AxisWithLengthBasisUtil(basis)
    origin = tuple(0 for _ in basis) if origin is None else origin
    origin = origin if isinstance(origin, tuple) else util.get_stacked_index(origin)
    return tuple(  # type: ignore[return-value]
        _wrap_distance(idx[ax] - origin[ax], util.shape[ax]) + origin[ax]
        if axes is None or ax in axes
        else idx[ax]
        for ax in range(util.ndim)
    )


def calculate_distances_along_path(
    basis: _ALB0Inv,
    path: np.ndarray[tuple[int, int], np.dtype[np.int_]],
    *,
    wrap_distances: bool = False,
) -> np.ndarray[tuple[int], np.dtype[np.int_]]:
    """
    calculate cumulative distances along the given path.

    Parameters
    ----------
    basis : _B0Inv
        basis which the path is through
    path : np.ndarray[tuple[int, int], np.dtype[np.int_]]
        path through the basis, _ND by int points
    wrap_distances : bool, optional
        wrap the distances into the first unit cell, by default False

    Returns
    -------
    np.ndarray[tuple[int], np.dtype[np.int_]]
    """
    out = path[:, :-1] - path[:, 1:]
    if wrap_distances:
        util = AxisWithLengthBasisUtil(basis)
        return np.array(  # type: ignore[no-any-return]
            [_wrap_distance(d, n) for (d, n) in zip(out, util.shape, strict=True)]
        )

    return out  # type:ignore[no-any-return]


def calculate_cumulative_x_distances_along_path(
    basis: _ALB0Inv,
    path: np.ndarray[tuple[int, int], np.dtype[np.int_]],
    *,
    wrap_distances: bool = False,
) -> np.ndarray[tuple[int], np.dtype[np.float_]]:
    """
    calculate the cumulative distances along the given path in the given basis.

    Parameters
    ----------
    basis : _B3d0Inv
        basis which the path is through
    path : np.ndarray[tuple[Literal[3], int], np.dtype[np.int_]]
        path, as a list of indexes in the basis, _ND by int points
    wrap_distances : bool, optional
        wrap the distances into the first unit cell, by default False

    Returns
    -------
    np.ndarray[tuple[int], np.dtype[np.float_]]
    """
    distances = calculate_distances_along_path(
        basis, path, wrap_distances=wrap_distances
    )

    util = AxisWithLengthBasisUtil(basis)
    x_distances = np.linalg.norm(
        np.tensordot(util.fundamental_dx, distances, axes=(0, 0)), axis=0
    )
    cum_distances = np.cumsum(x_distances)
    # Add back initial distance
    return np.insert(cum_distances, 0, 0)  # type: ignore[no-any-return]


def calculate_cumulative_k_distances_along_path(
    basis: _B3d0Inv,
    path: np.ndarray[tuple[Literal[3], int], np.dtype[np.int_]],
    *,
    wrap_distances: bool = False,
) -> np.ndarray[tuple[int], np.dtype[np.float_]]:
    """
    calculate the cumulative distances along the given path in the given basis.

    Parameters
    ----------
    basis : _B3d0Inv
        basis which the path is through
    path : np.ndarray[tuple[Literal[3], int], np.dtype[np.int_]]
        path, as a list of indexes in the basis
    wrap_distances : bool, optional
        wrap the distances into the first unit cell, by default False

    Returns
    -------
    np.ndarray[tuple[int], np.dtype[np.float_]]
    """
    (d0, d1, d2) = calculate_distances_along_path(
        basis, path, wrap_distances=wrap_distances  # type: ignore[arg-type]
    )
    util = AxisWithLengthBasisUtil(basis)
    x_distances = np.linalg.norm(
        d0[np.newaxis, :] * util.dk0[:, np.newaxis]
        + d1[np.newaxis, :] * util.dk1[:, np.newaxis]
        + d2[np.newaxis, :] * util.dk2[:, np.newaxis],
        axis=0,
    )
    cum_distances = np.cumsum(x_distances)
    # Add back initial distance
    return np.insert(cum_distances, 0, 0)  # type: ignore[no-any-return]


@overload
def get_x01_mirrored_index(idx: SingleStackedIndexLike) -> SingleStackedIndexLike:
    ...


@overload
def get_x01_mirrored_index(
    idx: ArrayStackedIndexLike[_S0Inv],
) -> ArrayStackedIndexLike[_S0Inv]:
    ...


def get_x01_mirrored_index(idx: StackedIndexLike) -> StackedIndexLike:
    """
    Mirror the coordinate idx about x0=x1.

    Parameters
    ----------
    basis : _B3d0Inv
        the basis to mirror in
    idx : tuple[int, int, int] | int
        The index to mirror

    Returns
    -------
    tuple[int, int, int] | int
        The mirrored index
    """
    arr = list(idx)
    arr[0] = idx[1]
    arr[1] = idx[0]
    return tuple(arr)  # type: ignore[return-value]


def get_single_point_basis(
    basis: _ALB0Inv,
) -> AxisWithLengthBasis[tuple[FundamentalPositionAxis[Literal[1], int], ...]]:
    """
    Get the basis with a single point in position space.

    Parameters
    ----------
    basis : _B3d0Inv
        initial basis
    _type : Literal[&quot;position&quot;, &quot;momentum&quot;]
        type of the final basis

    Returns
    -------
    _SPB|_SMB
        the single point basis in either position or momentum basis
    """
    return tuple(FundamentalPositionAxis3d(b.delta_x, 1) for b in basis)  # type: ignore[return-value]
