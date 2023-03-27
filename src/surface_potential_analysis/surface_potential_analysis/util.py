import datetime
from functools import wraps
from types import EllipsisType
from typing import Any, Callable, Literal, TypeVar

import numpy as np

F = TypeVar("F", bound=Callable[..., Any])


def timed(f: F) -> F:
    @wraps(f)
    def wrap(*args, **kw):  # type: ignore
        ts = datetime.datetime.now()
        result = f(*args, **kw)
        te = datetime.datetime.now()
        print(f"func: {f.__name__} took: {(te - ts).total_seconds()} sec")
        return result

    return wrap  # type: ignore


def slice_along_axis(
    slice_at_axis: slice | int, axis: int = -1
) -> tuple[EllipsisType | slice | int, ...] | tuple[slice | int, ...]:
    """
    Returns a slice such that the 1d slice provided by slice_inds, slices along the dimension provided.
    """
    from_end = False
    if axis < 0:  # choosing axis at the end
        from_end = True
        axis = -1 - axis
    # Pad the slice with slice(None)
    slice_padding = axis * (slice(None),)
    if from_end:
        return (Ellipsis, slice_at_axis) + slice_padding
    else:
        return slice_padding + (slice_at_axis,)


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
    Get a list of cumulative distances along a path ([x_coord, y_coord, z_coord]) given a grid of coordinates

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
    return np.insert(cum_distances, 0, 0)  # type:ignore
