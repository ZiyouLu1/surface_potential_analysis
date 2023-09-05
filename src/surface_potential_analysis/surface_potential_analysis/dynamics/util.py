from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Unpack

import numpy as np

from surface_potential_analysis.axis.axis import FundamentalAxis
from surface_potential_analysis.basis.util import BasisUtil

if TYPE_CHECKING:
    _S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])


def get_hop_shift(hop: int, ndim: int) -> tuple[int, ...]:
    """
    Given a hop index in ndim, get the hop shift vector.

    Parameters
    ----------
    hop : int
    ndim : int

    Returns
    -------
    tuple[int, ...]
    """
    util = BasisUtil(tuple(FundamentalAxis(3) for _ in range(ndim)))
    return tuple(x.item(hop) for x in util.nk_points)


def build_hop_operator(
    hop: int, shape: _S0Inv
) -> np.ndarray[tuple[Unpack[_S0Inv], Unpack[_S0Inv]], np.dtype[np.float_]]:
    """
    Given a hop index, build a hop operator in the given shape.

    Parameters
    ----------
    hop : int
        hop index
    shape : _S0Inv
        shape

    Returns
    -------
    np.ndarray[tuple[Unpack[_S0Inv], Unpack[_S0Inv]], np.dtype[np.real_]]
    """
    hop_shift = get_hop_shift(hop, len(shape))
    operator = np.identity(int(np.prod(shape))).reshape(*shape, *shape)
    return np.roll(operator, hop_shift, tuple(range(len(shape))))  # type: ignore[no-any-return]
