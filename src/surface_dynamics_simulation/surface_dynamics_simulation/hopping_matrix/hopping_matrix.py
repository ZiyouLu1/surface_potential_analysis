from __future__ import annotations

from typing import Generic, Literal, TypeVar

import numpy as np

_L0Inv = TypeVar("_L0Inv", bound=int)


class HoppingMatrix(
    np.ndarray[tuple[_L0Inv, _L0Inv, Literal[9]], np.dtype[np.float_]], Generic[_L0Inv]
):
    """
    Represents the matrix of hopping coefficients on a surfaces.

    The coefficients np.ndarray[tuple[_L0Inv, _L0Inv, Literal[9]], np.dtype[np.float_]]
    represent the total rate R[i,j,dx] from i to j with an offset of dx at the location i.

    dx is indexed such that np.unravel_multi_index(offset, (3,3), mode="wrap") gives the flat index
    for offsets (-1/+0/+1, -1/+0/+1)

    For example for a grid with i,j = {0,1}, ignoring tunnelling to neighboring unit cells
    - [0,0,0] = 0
    - [0,1,0] is the rate out from 0 in to 1
    - [1,0,0] is the rate out from 1 in to 0
    the overall rate of change of site 0 is then [1,0,0] - [0,1,0]
    ie the rate in from 1 minus the rate out from zero
    Note: [i,i,0] should always be zero (for now, maybe later we add coherent??)
    Note: all elements of R should be positive, as we are dealing with a rate of flow out
    from i, which depends on the value at i
    """

    def __init__(
        self, array: np.ndarray[tuple[_L0Inv, _L0Inv, Literal[9]], np.dtype[np.float_]]
    ) -> None:
        pass

    def __call__(self, x):  # type: ignore[no-untyped-def] # noqa: ANN204, D102, ANN001
        return x
