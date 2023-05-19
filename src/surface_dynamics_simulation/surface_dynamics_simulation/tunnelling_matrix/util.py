from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np

if TYPE_CHECKING:
    from .tunnelling_matrix import (
        TunnellingMatrix,
        TunnellingVector,
    )

    _L0Inv = TypeVar("_L0Inv", bound=int)
    _L1Inv = TypeVar("_L1Inv", bound=int)
    _L2Inv = TypeVar("_L2Inv", bound=int)
    _S0Inv = TypeVar("_S0Inv", bound=tuple[int, int, int])


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
    stacked = matrix["array"].reshape(*matrix["shape"], *matrix["shape"])
    array = stacked[0, 0, :, 0, 0, :]
    return {"shape": (1, 1, matrix["shape"][2]), "array": array}


def get_all_single_site_tunnelling_vectors(
    matrix: TunnellingMatrix[_S0Inv],
) -> list[TunnellingVector[_S0Inv]]:
    """
    Given a tunnelling matrix, get all tunnelling vectors located at the (0,0) site.

    Parameters
    ----------
    matrix : TunnellingMatrix[_S0Inv]

    Returns
    -------
    TunnellingMatrix[_S0Inv]
    """
    vectors: list[TunnellingVector[_S0Inv]] = []
    for n in range(matrix["shape"][2]):
        vector = np.zeros(matrix["shape"])
        vector[0, 0, n] = 1
        vectors.append({"shape": matrix["shape"], "vector": vector.flatten()})
    return vectors
