from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar

if TYPE_CHECKING:
    from .tunnelling_matrix import (
        TunnellingMatrix,
    )

    _L0Inv = TypeVar("_L0Inv", bound=int)
    _L1Inv = TypeVar("_L1Inv", bound=int)
    _L2Inv = TypeVar("_L2Inv", bound=int)


def get_single_site_tunnelling_matrix(
    matrix: TunnellingMatrix[tuple[_L0Inv, _L1Inv, _L2Inv]]
) -> TunnellingMatrix[tuple[Literal[1], Literal[1], _L2Inv]]:
    """
    Given a tunnelling matrix, get the matrix tunnelling only between a single site.

    Parameters
    ----------
    matrix : TunnellingMatrix[tuple[_L0Inv, _L1Inv, _L2Inv]]

    Returns
    -------
    TunnellingMatrix[tuple[Literal[1], Literal[1], _L2Inv]]
    """
    print(matrix["shape"])
    stacked = matrix["array"].reshape(*matrix["shape"], *matrix["shape"])
    array = stacked[0, 0, :, 0, 0, :]
    return {"shape": (1, 1, matrix["shape"][2]), "array": array}
