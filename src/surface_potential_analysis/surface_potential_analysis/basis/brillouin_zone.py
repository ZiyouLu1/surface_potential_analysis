from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar, Unpack, overload

import numpy as np

from surface_potential_analysis.basis.build import (
    position_basis_3d_from_resolution,
)
from surface_potential_analysis.basis.util import Basis3dUtil, BasisUtil

if TYPE_CHECKING:
    from surface_potential_analysis._types import (
        ArrayStackedIndexLike,
        ArrayStackedIndexLike3d,
        SingleStackedIndexLike,
        SingleStackedIndexLike3d,
        StackedIndexLike,
        StackedIndexLike3d,
    )
    from surface_potential_analysis.basis.basis import (
        Basis,
        Basis3d,
        FundamentalPositionBasis3d,
    )

    _B3d0Inv = TypeVar("_B3d0Inv", bound=Basis3d[Any, Any, Any])
    _B0Inv = TypeVar("_B0Inv", bound=Basis[Any])
_S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])


@overload
def fold_point_in_bragg_plane(
    bragg_point: SingleStackedIndexLike3d, coordinate: ArrayStackedIndexLike3d[_S0Inv]
) -> ArrayStackedIndexLike3d[_S0Inv]:
    ...


@overload
def fold_point_in_bragg_plane(
    bragg_point: SingleStackedIndexLike3d, coordinate: SingleStackedIndexLike3d
) -> SingleStackedIndexLike3d:
    ...


def fold_point_in_bragg_plane(
    bragg_point: SingleStackedIndexLike3d, coordinate: StackedIndexLike3d
) -> StackedIndexLike3d:
    """
    Given a bragg point and a StackedIndexLike fold the coordinates about the bragg plane.

    If the bragg_point corresponds to the closest brag plane outside the point
    this increments the brillouin zone by 1.
    If the bragg_point corresponds to the closest brag plane inside the point
    this increments the brillouin zone by 1.

    Parameters
    ----------
    bragg_point : SingleStackedIndexLike
        bragg point (point of symmetry corresponding to a bragg plane)
    coordinate : StackedIndexLike
        coordinate(s) to fold

    Returns
    -------
    StackedIndexLike
        folded coordinate(s)
    """
    # To fold a point about a bragg plane we
    # reflect it through the plane and then reflect it about
    # a plane parallel to this at the origin.
    # If we denote the plane by the normal n and point p, where p.n n = p
    # we find p is given by bragg_point / 2.
    # The two reflections are then equivalent to a translation a - 2p.
    return (
        coordinate[0] - bragg_point[0],
        coordinate[1] - bragg_point[1],
        coordinate[2] - bragg_point[2],
    )


def get_bragg_point_basis(
    basis: _B3d0Inv, *, n_bands: int = 1
) -> FundamentalPositionBasis3d[int, int, int]:
    """
    Get the basis for the bragg points, where the k_points are the bragg points of the given basis.

    Parameters
    ----------
    basis : _B3d0Inv
        parent basis
    n_bands : int, optional
        number of bands, by default 1

    Returns
    -------
    FundamentalPositionBasis3d[int, int, int]
    """
    width = 2 * n_bands + 1
    util = Basis3dUtil(basis)
    delta_x = (util.dx0, util.dx1, util.dx2)
    # we want a basis where dk_1 = delta_k_original
    # Since dk = 2 * pi / delta_x we need to decrease delta_x by a factor of util.ni
    # as this will increase dk to the intended value.
    # This is equivalent to delta_x = (util.dx0, util.dx1, util.dx2)
    return position_basis_3d_from_resolution((width, width, width), delta_x)


def get_all_brag_point_index(
    basis: _B3d0Inv, n_bands: int = 1
) -> ArrayStackedIndexLike3d[tuple[int]]:
    """
    Given a basis in 3D, get the stacked index of the brag points in the first n_bands.

    Parameters
    ----------
    basis : _B3d0Inv
    n_bands : int, optional
        number of bands of the brag points, by default 1

    Returns
    -------
    ArrayStackedIndexLike[tuple[int]]
        Index of each of the brag points, using the nk_points convention for the ArrayStackedIndexLike
    """
    mock_basis = position_basis_3d_from_resolution(
        (2 * n_bands + 1, 2 * n_bands + 1, 2 * n_bands + 1)
    )
    nk_points = Basis3dUtil(mock_basis).fundamental_nk_points
    return (
        nk_points[0] * basis[0].n,
        nk_points[1] * basis[1].n,
        nk_points[2] * basis[2].n,
    )


def get_all_brag_point(
    basis: _B3d0Inv, *, n_bands: int = 1
) -> np.ndarray[tuple[Literal[3], int], np.dtype[np.float_]]:
    """
    Given a basis in 3D, get the coordinates of the brag points in the first n_bands.

    Parameters
    ----------
    basis : _B3d0Inv
    n_bands : int, optional
        number of bands of the brag points, by default 1

    Returns
    -------
    np.ndarray[tuple[Literal[3], int], np.dtype[np.float_]]
        Array of each of the brag points, using the nk_points convention
    """
    bragg_point_basis = get_bragg_point_basis(basis, n_bands=n_bands)
    return Basis3dUtil(bragg_point_basis).fundamental_k_points


@overload
def get_bragg_plane_distance(
    bragg_point: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]],
    point: np.ndarray[tuple[Literal[3], Unpack[_S0Inv]], np.dtype[np.float_]],
) -> np.ndarray[_S0Inv, np.dtype[np.float_]]:
    ...


@overload
def get_bragg_plane_distance(
    bragg_point: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]],
    point: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]],
) -> np.float_:
    ...


def get_bragg_plane_distance(
    bragg_point: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]],
    point: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    | np.ndarray[tuple[Literal[3], Unpack[_S0Inv]], np.dtype[np.float_]],
) -> np.float_ | np.ndarray[_S0Inv, np.dtype[np.float_]]:
    """
    Get the distance from the bragg plane for the given bragg_point.

    Parameters
    ----------
    bragg_point : np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
        bragg point (point of symmetry corresponding to a bragg plane)
    point : np.ndarray[tuple[Literal[3]], np.dtype[np.float_]] | np.ndarray[tuple[Literal[3], Unpack[_S0Inv]], np.dtype[np.float_]]

    Returns
    -------
    np.float_ | np.ndarray[tuple[int, ...], np.dtype[np.float_]]
    """
    bragg_point_norm = np.linalg.norm(bragg_point, axis=0)
    normalized_bragg_point = np.array(bragg_point) / bragg_point_norm
    # d = n.(a-p) = n.a - 0.5 * n.d where p = d/2 is the point on the bragg plane
    return np.tensordot(normalized_bragg_point, point, axes=(0, 0)) - (  # type: ignore[no-any-return]
        bragg_point_norm / 2
    )


def get_closest_interior_bragg_plane(
    basis: _B3d0Inv, coordinate: ArrayStackedIndexLike3d
) -> None:
    np.zeros_like(coordinate[0])


def increment_brillouin_zone(basis: _B3d0Inv, coordinate: SingleStackedIndexLike3d):
    raise NotImplementedError
    # To fold a point into the next brillouin zone we reflect it
    # in the closest bragg plane outside the point and then reflect it about
    # a plane parallel to this at the origin.


def _get_decrement_tolerance(basis: _B3d0Inv) -> float:
    r_tol = 1e-5
    util = Basis3dUtil(basis)
    return np.min(np.linalg.norm([util.dk0, util.dk1, util.dk2], axis=1)) * r_tol


_NInv = TypeVar("_NInv", bound=int)
StackedIndexLikeArray = np.ndarray[tuple[_NInv, Unpack[_S0Inv]], np.dtype[np.int_]]


@overload
def decrement_brillouin_zone_3d(
    basis: _B3d0Inv, coordinate: ArrayStackedIndexLike3d[_S0Inv]
) -> ArrayStackedIndexLike3d[_S0Inv]:
    ...


@overload
def decrement_brillouin_zone_3d(
    basis: _B3d0Inv, coordinate: SingleStackedIndexLike3d
) -> SingleStackedIndexLike3d:
    ...


def decrement_brillouin_zone_3d(
    basis: _B3d0Inv, coordinate: StackedIndexLike3d
) -> StackedIndexLike3d:
    """
    Given a basis, and a set of coordinates, decrement the brillouin zone of each coordinate.

    Note if the coordinate is in the 1st (ie lowest) zone it wil remain in the same location

    Parameters
    ----------
    basis : _B3d0Inv
    coordinate : StackedIndexLike

    Returns
    -------
    StackedIndexLike
        coordinates in the min(1, n-1) th brillouin zone
    """
    tolerance = _get_decrement_tolerance(basis)
    # Transpose used - we want the new axis to appear as the last axis not the first
    out = np.atleast_2d(np.transpose(coordinate)).T
    coordinate_points = Basis3dUtil(basis).get_k_points_at_index(tuple(out))  # type: ignore[arg-type]
    bragg_points = get_all_brag_point(basis, n_bands=1)

    closest_points = np.zeros_like(out[0], dtype=np.int_)
    distances = np.linalg.norm(coordinate_points, axis=0)
    # To fold a point into the previous brillouin zone we reflect it
    # in the closest bragg plane inside the point and then reflect it about
    # a plane parallel to this at the origin.
    for i, bragg_point in enumerate(np.array(bragg_points)[:, 1:].T):
        new_distances = get_bragg_plane_distance(bragg_point, coordinate_points)  # type: ignore[arg-type,var-annotated]
        is_closer = np.logical_and(new_distances > tolerance, new_distances < distances)
        closest_points[is_closer] = i + 1
        distances[is_closer] = new_distances[is_closer]

    bragg_point_index = get_all_brag_point_index(basis, n_bands=1)
    for i, bragg_point in enumerate(np.array(bragg_point_index).T):
        should_fold = closest_points == i
        folded = fold_point_in_bragg_plane(bragg_point, tuple(out[:, should_fold]))  # type: ignore[arg-type,var-annotated]
        out[:, should_fold] = folded

    if isinstance(coordinate[0], np.ndarray):
        # For 0D arrays we need to drop the additional axis here
        old_shape = coordinate[0].shape
        return (
            out[0].reshape(old_shape),
            out[1].reshape(old_shape),
            out[2].reshape(old_shape),
        )
    return (out.item(0), out.item(1), out.item(2))


@overload
def decrement_brillouin_zone(
    basis: _B0Inv, coordinate: ArrayStackedIndexLike[_S0Inv]
) -> ArrayStackedIndexLike[_S0Inv]:
    ...


@overload
def decrement_brillouin_zone(
    basis: _B0Inv, coordinate: SingleStackedIndexLike
) -> SingleStackedIndexLike:
    ...


def decrement_brillouin_zone(
    basis: _B0Inv, coordinate: StackedIndexLike
) -> StackedIndexLike:
    """
    Given a basis, and a set of coordinates, decrement the brillouin zone of each coordinate.

    Note if the coordinate is in the 1st (ie lowest) zone it wil remain in the same location

    Parameters
    ----------
    basis : _B3d0Inv
    coordinate : StackedIndexLike

    Returns
    -------
    StackedIndexLike
        coordinates in the min(1, n-1) th brillouin zone
    """
    tolerance = _get_decrement_tolerance(basis)
    # Transpose used - we want the new axis to appear as the last axis not the first
    out = np.atleast_2d(np.transpose(coordinate)).T
    coordinate_points = BasisUtil(basis).get_k_points_at_index(tuple(out))  # type: ignore[arg-type]
    bragg_points = get_all_brag_point(basis, n_bands=1)

    closest_points = np.zeros_like(out[0], dtype=np.int_)
    distances = np.linalg.norm(coordinate_points, axis=0)
    # To fold a point into the previous brillouin zone we reflect it
    # in the closest bragg plane inside the point and then reflect it about
    # a plane parallel to this at the origin.
    for i, bragg_point in enumerate(np.array(bragg_points)[:, 1:].T):
        new_distances = get_bragg_plane_distance(bragg_point, coordinate_points)  # type: ignore[arg-type,var-annotated]
        is_closer = np.logical_and(new_distances > tolerance, new_distances < distances)
        closest_points[is_closer] = i + 1
        if np.sometrue(is_closer):
            print(bragg_point)
            print(is_closer.shape, closest_points.shape)
            print(tuple(out[:, is_closer]))
            print(new_distances[is_closer])
        distances[is_closer] = new_distances[is_closer]

    bragg_point_index = get_all_brag_point_index(basis, n_bands=1)
    for i, bragg_point in enumerate(np.array(bragg_point_index).T):
        should_fold = closest_points == i
        folded = fold_point_in_bragg_plane(bragg_point, tuple(out[:, should_fold]))  # type: ignore[arg-type,var-annotated]
        if np.sometrue(should_fold):
            print(
                i,
                bragg_point,
                tuple(out[:, should_fold]),
                (out[0, should_fold], out[1, should_fold], out[2, should_fold]),
                folded,
                out.shape,
            )
        out[:, should_fold] = folded

    if isinstance(coordinate[0], np.ndarray):
        # For 0D arrays we need to drop the additional axis here
        old_shape = coordinate[0].shape
        return (
            out[0].reshape(old_shape),
            out[1].reshape(old_shape),
            out[2].reshape(old_shape),
        )
    return (out.item(0), out.item(1), out.item(2))
