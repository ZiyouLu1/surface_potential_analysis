from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, ParamSpec, TypeVar

import numpy as np

from surface_potential_analysis._types import _IntLike_co

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import EllipsisType

    P = ParamSpec("P")
    R = TypeVar("R")


def slice_along_axis(
    slice_at_axis: slice | _IntLike_co | None, axis: int = -1
) -> (
    tuple[EllipsisType | slice | _IntLike_co | None, ...]
    | tuple[slice | _IntLike_co | None, ...]
):
    """Return a slice such that the 1d slice provided by slice_at_axis, slices along the dimension provided."""
    from_end = False
    if axis < 0:  # choosing axis at the end
        from_end = True
        axis = -1 - axis
    # Pad the slice with slice(None)
    slice_padding = axis * (slice(None),)
    if from_end:
        return (Ellipsis, slice_at_axis, *slice_padding)

    return (*slice_padding, slice_at_axis)


_AX0Inv = TypeVar("_AX0Inv", bound=tuple[_IntLike_co, ...])


def slice_ignoring_axes(
    old_slice: Sequence[slice | _IntLike_co | None], axes: _AX0Inv
) -> tuple[slice | _IntLike_co | None, ...]:
    """
    Given a slice, insert slice(None) everywhere given in axes.

    Parameters
    ----------
    slice : list[slice  |  _IntLike_co  |  None]
        _description_
    axes : tuple[_IntLike_co]
        _description_

    Returns
    -------
    list[slice | _IntLike_co | None]
        _description_
    """
    old_slice = list(old_slice)
    for axis in sorted(axes):
        old_slice.insert(axis, slice(None))
    return tuple(old_slice)


_LInv = TypeVar("_LInv", bound=int)

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)


def calculate_cumulative_distances_along_path(
    path: np.ndarray[tuple[Literal[3], _LInv], np.dtype[np.int_]],
    coordinates: np.ndarray[
        tuple[Literal[3], _L0Inv, _L1Inv, _L2Inv], np.dtype[np.float_]
    ],
) -> np.ndarray[tuple[_LInv], np.dtype[np.float_]]:
    """
    Get a list of cumulative distances along a path ([x_coord, y_coord, z_coord]) given a grid of coordinates.

    Parameters
    ----------
    path : np.ndarray[tuple[Literal[3], _L], np.dtype[np.float_]]
        Path for which to calculate the distance along (as a list for each coordinate)
    coordinates : np.ndarray[tuple[_L0, _L1, _L2], np.dtype[np.float_]]
        Coordinate grid, with the same dimension as the coordinates given in the path

    Returns
    -------
    np.ndarray[tuple[_L], np.dtype[np.float_]]
        The distance from the starting coordinate along the given path
    """
    path_coordinates = coordinates[:, *path]
    distances = np.linalg.norm(
        path_coordinates[:, :-1] - path_coordinates[:, 1:], axis=0
    )
    cum_distances = np.cumsum(distances)
    # Add back initial distance
    return np.insert(cum_distances, 0, 0)  # type: ignore[no-any-return]


_SInv = TypeVar("_SInv", bound=tuple[Any])


Measure = Literal["real", "imag", "abs", "angle"]


def get_measured_data(
    data: np.ndarray[_SInv, np.dtype[np.complex_]],
    measure: Measure,
) -> np.ndarray[_SInv, np.dtype[np.float_]]:
    """
    Transform data with the given measure.

    Parameters
    ----------
    data : np.ndarray[_SInv, np.dtype[np.complex_]]
    measure : Literal[&quot;real&quot;, &quot;imag&quot;, &quot;abs&quot;, &quot;angle&quot;]


    Returns
    -------
    np.ndarray[_SInv, np.dtype[np.float_]]
    """
    match measure:
        case "real":
            return np.real(data)  # type: ignore[no-any-return]
        case "imag":
            return np.imag(data)  # type: ignore[no-any-return]
        case "abs":
            return np.abs(data)  # type: ignore[no-any-return]
        case "angle":
            return np.unwrap(np.angle(data))  # type: ignore[no-any-return]
