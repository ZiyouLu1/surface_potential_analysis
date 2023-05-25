from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, Unpack, overload

import numpy as np

from surface_potential_analysis.basis.basis import (
    FundamentalPositionBasis,
)
from surface_potential_analysis.basis.basis_like import (
    BasisLike,
    BasisVector,
)
from surface_potential_analysis.basis.util import (
    BasisUtil,
)

from .basis_config import BasisConfig
from .conversion import get_rotated_basis_config

if TYPE_CHECKING:
    from surface_potential_analysis._types import (
        ArrayFlatIndexLike,
        ArrayIndexLike,
        ArrayStackedIndexLike,
        FlatIndexLike,
        IndexLike,
        SingleFlatIndexLike,
        SingleIndexLike,
        SingleStackedIndexLike,
        StackedIndexLike,
        _IntLike_co,
    )

_BC0Inv = TypeVar("_BC0Inv", bound=BasisConfig[Any, Any, Any])


_S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])


_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)

_LF0Inv = TypeVar("_LF0Inv", bound=int)
_LF1Inv = TypeVar("_LF1Inv", bound=int)
_LF2Inv = TypeVar("_LF2Inv", bound=int)

_BX0Inv = TypeVar("_BX0Inv", bound=BasisLike[Any, Any])
_BX1Inv = TypeVar("_BX1Inv", bound=BasisLike[Any, Any])
_BX2Inv = TypeVar("_BX2Inv", bound=BasisLike[Any, Any])


# ruff: noqa: D102
class BasisConfigUtil(Generic[_BC0Inv]):
    """A class to help with the manipulation of basis configs."""

    _config: _BC0Inv

    def __init__(self, config: _BC0Inv) -> None:
        self._config = config

    @property
    def config(self) -> _BC0Inv:
        return self._config

    @cached_property
    def volume(self) -> float:
        out = np.dot(self.delta_x0, np.cross(self.delta_x1, self.delta_x2))
        assert out != 0
        return out  # type: ignore[no-any-return]

    @cached_property
    def reciprocal_volume(self) -> float:
        out = np.dot(self.dk0, np.cross(self.dk1, self.dk2))
        assert out != 0
        return out  # type: ignore[no-any-return]

    @property
    def nk_points(self) -> ArrayStackedIndexLike[tuple[int]]:
        x0t, x1t, x2t = np.meshgrid(
            self.x0_basis.nk_points,  # type: ignore[misc]
            self.x1_basis.nk_points,  # type: ignore[misc]
            self.x2_basis.nk_points,  # type: ignore[misc]
            indexing="ij",
        )
        return (x0t.ravel(), x1t.ravel(), x2t.ravel())

    @property
    def fundamental_nk_points(self) -> ArrayStackedIndexLike[tuple[int]]:
        x0t, x1t, x2t = np.meshgrid(
            self.x0_basis.fundamental_nk_points,  # type: ignore[misc]
            self.x1_basis.fundamental_nk_points,  # type: ignore[misc]
            self.x2_basis.fundamental_nk_points,  # type: ignore[misc]
            indexing="ij",
        )
        return (x0t.ravel(), x1t.ravel(), x2t.ravel())

    def get_k_points_at_index(
        self, idx: ArrayIndexLike[_S0Inv]
    ) -> np.ndarray[tuple[Literal[3], Unpack[_S0Inv]], np.dtype[np.float_]]:
        idx = idx if isinstance(idx, tuple) else self.get_stacked_index(idx)

        nk_points = np.asarray(idx)[:, np.newaxis, :]
        basis_vectors = np.array([self.dk0, self.dk1, self.dk2])[:, :, np.newaxis]
        return np.sum(basis_vectors * nk_points, axis=0)  # type: ignore[no-any-return]

    @property
    def k_points(
        self,
    ) -> np.ndarray[tuple[Literal[3], int], np.dtype[np.float_]]:
        return self.get_k_points_at_index(self.nk_points)

    @property
    def fundamental_k_points(
        self,
    ) -> np.ndarray[tuple[Literal[3], int], np.dtype[np.float_]]:
        nk_points = np.asarray(self.fundamental_nk_points)[:, np.newaxis, :]
        basis_vectors = np.array(
            [self.fundamental_dk0, self.fundamental_dk1, self.fundamental_dk2]
        )[:, :, np.newaxis]
        return np.sum(basis_vectors * nk_points, axis=0)  # type: ignore[no-any-return]

    @property
    def nx_points(self) -> ArrayStackedIndexLike[tuple[int]]:
        x0t, x1t, x2t = np.meshgrid(
            self.x0_basis.nx_points,  # type: ignore[misc]
            self.x1_basis.nx_points,  # type: ignore[misc]
            self.x2_basis.nx_points,  # type: ignore[misc]
            indexing="ij",
        )
        return (x0t.ravel(), x1t.ravel(), x2t.ravel())

    @property
    def fundamental_nx_points(self) -> ArrayStackedIndexLike[tuple[int]]:
        x0t, x1t, x2t = np.meshgrid(
            self.x0_basis.fundamental_nx_points,  # type: ignore[misc]
            self.x1_basis.fundamental_nx_points,  # type: ignore[misc]
            self.x2_basis.fundamental_nx_points,  # type: ignore[misc]
            indexing="ij",
        )
        return (x0t.ravel(), x1t.ravel(), x2t.ravel())

    def get_x_points_at_index(
        self, idx: ArrayIndexLike[_S0Inv]
    ) -> np.ndarray[tuple[Literal[3], Unpack[_S0Inv]], np.dtype[np.float_]]:
        idx = idx if isinstance(idx, tuple) else self.get_stacked_index(idx)

        nk_points = np.asarray(idx)[:, np.newaxis, :]
        basis_vectors = np.array([self.dx0, self.dx1, self.dx2])[:, :, np.newaxis]
        return np.tensordot(basis_vectors, nk_points, axes=0)  # type: ignore[no-any-return]

    @property
    def x_points(
        self,
    ) -> np.ndarray[tuple[Literal[3], int], np.dtype[np.float_]]:
        return self.get_x_points_at_index(self.nx_points)

    @property
    def fundamental_x_points(
        self,
    ) -> np.ndarray[tuple[Literal[3], int], np.dtype[np.float_]]:
        nx_points = np.asarray(self.fundamental_nx_points)[:, np.newaxis, :]
        basis_vectors = np.array(
            [self.fundamental_dx0, self.fundamental_dx1, self.fundamental_dx2]
        )[:, :, np.newaxis]
        return np.sum(basis_vectors * nx_points, axis=0)  # type: ignore[no-any-return]

    @cached_property
    def x0_basis(
        self: BasisConfigUtil[tuple[BasisLike[_LF0Inv, _L0Inv], _BX1Inv, _BX2Inv]]
    ) -> BasisUtil[_LF0Inv, _L0Inv]:
        return BasisUtil(self._config[0])

    @property
    def fundamental_n0(
        self: BasisConfigUtil[tuple[BasisLike[_LF0Inv, _L0Inv], _BX1Inv, _BX2Inv]]
    ) -> _LF0Inv:
        return self.x0_basis.fundamental_n  # type: ignore[misc]

    @property
    def n0(
        self: BasisConfigUtil[tuple[BasisLike[_LF0Inv, _L0Inv], _BX1Inv, _BX2Inv]]
    ) -> _L0Inv:
        return self.x0_basis.n

    @property
    def delta_x0(self) -> BasisVector:
        return self.x0_basis.delta_x  # type: ignore[misc]

    @property
    def fundamental_delta_x0(self) -> BasisVector:
        return self.delta_x0

    @cached_property
    def dx0(self) -> BasisVector:
        return self.x0_basis.dx  # type: ignore[misc]

    @cached_property
    def fundamental_dx0(self) -> BasisVector:
        return self.x0_basis.delta_x  # type: ignore[misc]

    @property
    def delta_k0(self) -> BasisVector:
        return self.n0 * self.dk0  # type: ignore[no-any-return, misc]

    @cached_property
    def fundamental_delta_k0(self) -> BasisVector:
        return self.fundamental_n0 * self.dk0  # type: ignore[no-any-return,misc]

    @cached_property
    def dk0(self) -> BasisVector:
        # See https://physics.stackexchange.com/questions/340860/reciprocal-lattice-in-2d
        return (  # type: ignore[no-any-return]
            2 * np.pi * np.cross(self.delta_x1, self.delta_x2) / self.volume
        )

    @property
    def fundamental_dk0(self) -> BasisVector:
        return self.dk0

    @cached_property
    def x1_basis(
        self: BasisConfigUtil[tuple[_BX0Inv, BasisLike[_LF1Inv, _L1Inv], _BX2Inv]]
    ) -> BasisUtil[_LF1Inv, _L1Inv]:
        return BasisUtil(self._config[1])

    @property
    def fundamental_n1(
        self: BasisConfigUtil[tuple[_BX0Inv, BasisLike[_LF1Inv, _L1Inv], _BX2Inv]]
    ) -> _LF1Inv:
        return self.x1_basis.fundamental_n

    @property
    def n1(
        self: BasisConfigUtil[tuple[_BX0Inv, BasisLike[_LF1Inv, _L1Inv], _BX2Inv]]
    ) -> _L1Inv:
        return self.x1_basis.n

    @property
    def delta_x1(self) -> BasisVector:
        return self.x1_basis.delta_x  # type: ignore[misc]

    @property
    def fundamental_delta_x1(self) -> BasisVector:
        return self.delta_x1

    @cached_property
    def dx1(self) -> BasisVector:
        return self.x1_basis.dx  # type: ignore[misc]

    @cached_property
    def fundamental_dx1(self) -> BasisVector:
        return self.x1_basis.fundamental_dx  # type: ignore[misc]

    @property
    def delta_k1(self) -> BasisVector:
        return self.n1 * self.dk1  # type: ignore[misc,no-any-return]

    @cached_property
    def fundamental_delta_k1(self) -> BasisVector:
        return self.fundamental_n1 * self.dk1  # type: ignore[misc,no-any-return]

    @cached_property
    def dk1(self) -> BasisVector:
        # See https://physics.stackexchange.com/questions/340860/reciprocal-lattice-in-2d
        out = 2 * np.pi * np.cross(self.delta_x2, self.delta_x0) / self.volume
        return out  # type: ignore[no-any-return]  # noqa: RET504

    @property
    def fundamental_dk1(self) -> BasisVector:
        return self.dk1

    @cached_property
    def x2_basis(
        self: BasisConfigUtil[tuple[_BX0Inv, _BX1Inv, BasisLike[_LF2Inv, _L2Inv]]]
    ) -> BasisUtil[_LF2Inv, _L2Inv]:
        return BasisUtil(self._config[2])

    @property
    def fundamental_n2(
        self: BasisConfigUtil[tuple[_BX0Inv, _BX1Inv, BasisLike[_LF2Inv, _L2Inv]]]
    ) -> int:
        return self.x2_basis.fundamental_n

    @property
    def n2(
        self: BasisConfigUtil[tuple[_BX0Inv, _BX1Inv, BasisLike[_LF2Inv, _L2Inv]]]
    ) -> int:
        return self.x2_basis.n

    @property
    def delta_x2(self) -> BasisVector:
        return self.x2_basis.delta_x  # type: ignore[misc]

    @property
    def fundamental_delta_x2(self) -> BasisVector:
        return self.delta_x2

    @cached_property
    def dx2(self) -> BasisVector:
        return self.x2_basis.dx  # type: ignore[misc]

    @cached_property
    def fundamental_dx2(self) -> BasisVector:
        return self.x2_basis.fundamental_dx  # type: ignore[misc]

    @property
    def delta_k2(self) -> BasisVector:
        return self.n2 * self.dk2  # type: ignore[misc,no-any-return]

    @cached_property
    def fundamental_delta_k2(self) -> BasisVector:
        return self.fundamental_n2 * self.dk2  # type: ignore[misc,no-any-return]

    @cached_property
    def dk2(self) -> BasisVector:
        # See https://physics.stackexchange.com/questions/340860/reciprocal-lattice-in-2d
        return (  # type: ignore[no-any-return]
            2 * np.pi * np.cross(self.delta_x0, self.delta_x1) / self.volume
        )

    @property
    def fundamental_dk2(self) -> BasisVector:
        return self.dk2

    @cached_property
    def shape(self) -> tuple[int, int, int]:
        return (self.x0_basis.n, self.x1_basis.n, self.x2_basis.n)  # type: ignore[misc]

    @property
    def size(self) -> int:
        return np.prod(self.shape)  # type: ignore[return-value]

    @cached_property
    def fundamental_shape(self) -> tuple[int, int, int]:
        return (self.x0_basis.fundamental_n, self.x1_basis.fundamental_n, self.x2_basis.fundamental_n)  # type: ignore[misc]

    def __len__(self) -> int:
        return int(np.prod(self.shape))

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


def get_fundamental_projected_k_points(
    basis: _BC0Inv,
    axis: Literal[0, 1, 2, -1, -2, -3],
) -> np.ndarray[tuple[Literal[2], int, int], np.dtype[np.float_]]:
    """
    Get a grid of points projected perpendicular to the given basis axis.

    This throws away the componet of the cooridnate grid in the direction
    parallel to axis.

    Parameters
    ----------
    basis : _BC0Inv
    axis : Literal[0, 1, 2,-1, -2, -3]
        The index along the axis to take the coordinates from

    Returns
    -------
    np.ndarray[tuple[Literal[2], int, int], np.dtype[np.float_]]
        The coordinates in the plane perpendicular to axis
    """
    rotated = get_rotated_basis_config(basis, axis)  # type: ignore[var-annotated,arg-type]
    util = BasisConfigUtil(rotated)
    return util.fundamental_k_points.reshape(3, *util.fundamental_shape)[0:2,]


def get_fundamental_projected_x_points(
    basis: _BC0Inv,
    axis: Literal[0, 1, 2, -1, -2, -3],
) -> np.ndarray[tuple[Literal[2], int, int, int], np.dtype[np.float_]]:
    """
    Get a grid of points projected perpendicular to the given basis axis, at the given index along this axis.

    This throws away the componet of the cooridnate grid in the direction
    parallel to axis.

    Parameters
    ----------
    basis : _BC0Inv
    axis : Literal[0, 1, 2,-1, -2, -3]
        The index along the axis to take the coordinates from

    Returns
    -------
    np.ndarray[tuple[Literal[2], int, int], np.dtype[np.float_]]
        The coordinates in the plane perpendicular to axis, as a list of [x_coords, y_coords, z_coords]
    """
    rotated = get_rotated_basis_config(basis, axis, np.array([0, 0, 1]))  # type: ignore[var-annotated,arg-type]
    util = BasisConfigUtil(rotated)
    return util.fundamental_x_points.reshape(3, *util.fundamental_shape)[0:2]


@overload
def _wrap_distance(distance: _IntLike_co, length: int) -> int:
    ...


@overload
def _wrap_distance(
    distance: np.ndarray[_S0Inv, np.dtype[np.int_]], length: int
) -> np.ndarray[_S0Inv, np.dtype[np.int_]]:
    ...


def _wrap_distance(distance: Any, length: int) -> Any:
    return np.subtract(np.mod(np.add(distance, length // 2), length), length // 2)


@overload
def wrap_index_around_origin_x01(
    basis: _BC0Inv, idx: SingleStackedIndexLike, origin_idx: SingleIndexLike = (0, 0, 0)
) -> SingleStackedIndexLike:
    ...


@overload
def wrap_index_around_origin_x01(
    basis: _BC0Inv, idx: SingleFlatIndexLike, origin_idx: SingleIndexLike = (0, 0, 0)
) -> SingleStackedIndexLike:
    ...


@overload
def wrap_index_around_origin_x01(
    basis: _BC0Inv,
    idx: ArrayStackedIndexLike[_S0Inv],
    origin_idx: SingleIndexLike = (0, 0, 0),
) -> ArrayStackedIndexLike[_S0Inv]:
    ...


@overload
def wrap_index_around_origin_x01(
    basis: _BC0Inv,
    idx: ArrayFlatIndexLike[_S0Inv],
    origin_idx: SingleIndexLike = (0, 0, 0),
) -> ArrayStackedIndexLike[_S0Inv]:
    ...


def wrap_index_around_origin_x01(
    basis: _BC0Inv,
    idx: StackedIndexLike | FlatIndexLike,
    origin_idx: SingleIndexLike = (0, 0, 0),
) -> StackedIndexLike:
    """
    Given an index or list of indexes in stacked form, find the equivalent index closest to the point origin_idx.

    Parameters
    ----------
    basis : _BC0Inv
    idx : StackedIndexLike | FlatIndexLike
    origin_idx : StackedIndexLike | FlatIndexLike, optional
        origin to wrap around, by default (0, 0, 0)

    Returns
    -------
    StackedIndexLike
    """
    util = BasisConfigUtil(basis)
    idx = idx if isinstance(idx, tuple) else util.get_stacked_index(idx)
    origin_idx = (
        origin_idx
        if isinstance(origin_idx, tuple)
        else util.get_stacked_index(origin_idx)
    )
    (n0, n1, _) = util.shape
    return (  # type: ignore[return-value]
        _wrap_distance(idx[0] - origin_idx[0], n0) + origin_idx[0],
        _wrap_distance(idx[1] - origin_idx[1], n1) + origin_idx[1],
        idx[2],
    )


def calculate_distances_along_path(
    basis: _BC0Inv,
    path: np.ndarray[tuple[Literal[3], int], np.dtype[np.int_]],
    *,
    wrap_distances: bool = False,
) -> np.ndarray[tuple[int], np.dtype[np.int_]]:
    """
    calculate cumulative distances along the given path.

    Parameters
    ----------
    basis : _BC0Inv
        basis which the path is through
    path : np.ndarray[tuple[Literal[3], int], np.dtype[np.int_]]
        path through the basis
    wrap_distances : bool, optional
        wrap the distances into the first unit cell, by default False

    Returns
    -------
    np.ndarray[tuple[int], np.dtype[np.int_]]
    """
    (d0, d1, d2) = path[:, :-1] - path[:, 1:]
    if wrap_distances:
        util = BasisConfigUtil(basis)
        d0 = _wrap_distance(d0, util.shape[0])
        d1 = _wrap_distance(d1, util.shape[1])
        d2 = _wrap_distance(d2, util.shape[2])

    return np.array([d0, d1, d2])  # type:ignore[no-any-return]


def calculate_cumulative_x_distances_along_path(
    basis: _BC0Inv,
    path: np.ndarray[tuple[Literal[3], int], np.dtype[np.int_]],
    *,
    wrap_distances: bool = False,
) -> np.ndarray[tuple[int], np.dtype[np.float_]]:
    """
    calculate the cumulative distances along the given path in the given basis.

    Parameters
    ----------
    basis : _BC0Inv
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
        basis, path, wrap_distances=wrap_distances
    )
    util = BasisConfigUtil(basis)
    x_distances = np.linalg.norm(
        d0[np.newaxis, :] * util.fundamental_dx0[:, np.newaxis]
        + d1[np.newaxis, :] * util.fundamental_dx1[:, np.newaxis]
        + d2[np.newaxis, :] * util.fundamental_dx2[:, np.newaxis],
        axis=0,
    )
    cum_distances = np.cumsum(x_distances)
    # Add back initial distance
    return np.insert(cum_distances, 0, 0)  # type: ignore[no-any-return]


def calculate_cumulative_k_distances_along_path(
    basis: _BC0Inv,
    path: np.ndarray[tuple[Literal[3], int], np.dtype[np.int_]],
    *,
    wrap_distances: bool = False,
) -> np.ndarray[tuple[int], np.dtype[np.float_]]:
    """
    calculate the cumulative distances along the given path in the given basis.

    Parameters
    ----------
    basis : _BC0Inv
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
        basis, path, wrap_distances=wrap_distances
    )
    util = BasisConfigUtil(basis)
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
def get_x01_mirrored_index(
    basis: _BC0Inv, idx: SingleStackedIndexLike
) -> SingleStackedIndexLike:
    ...


@overload
def get_x01_mirrored_index(basis: _BC0Inv, idx: SingleFlatIndexLike) -> np.int_:
    ...


@overload
def get_x01_mirrored_index(
    basis: _BC0Inv, idx: ArrayStackedIndexLike[_S0Inv]
) -> ArrayStackedIndexLike[_S0Inv]:
    ...


@overload
def get_x01_mirrored_index(
    basis: _BC0Inv, idx: ArrayFlatIndexLike[_S0Inv]
) -> ArrayFlatIndexLike[_S0Inv]:
    ...


def get_x01_mirrored_index(basis: _BC0Inv, idx: IndexLike) -> IndexLike:
    """
    Mirror the coordinate idx about x0=x1.

    Parameters
    ----------
    basis : _BC0Inv
        the basis to mirror in
    idx : tuple[int, int, int] | int
        The index to mirror

    Returns
    -------
    tuple[int, int, int] | int
        The mirrored index
    """
    util = BasisConfigUtil(basis)
    idx = idx if isinstance(idx, tuple) else util.get_stacked_index(idx)
    mirrored: StackedIndexLike = (idx[1], idx[0], idx[2])  # type: ignore[assignment]
    return mirrored if isinstance(idx, tuple) else util.get_flat_index(mirrored)


def get_single_point_basis(
    basis: _BC0Inv,
) -> BasisConfig[
    FundamentalPositionBasis[Literal[1]],
    FundamentalPositionBasis[Literal[1]],
    FundamentalPositionBasis[Literal[1]],
]:
    """
    Get the basis with a single point in position space.

    Parameters
    ----------
    basis : _BC0Inv
        initial basis
    _type : Literal[&quot;position&quot;, &quot;momentum&quot;]
        type of the final basis

    Returns
    -------
    _SPB|_SMB
        the single point basis in either position or momentum basis
    """
    return tuple(FundamentalPositionBasis(b.delta_x, 1) for b in basis)  # type: ignore[return-value]
