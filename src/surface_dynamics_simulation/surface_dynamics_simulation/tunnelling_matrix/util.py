from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np

if TYPE_CHECKING:
    from .tunnelling_matrix import (
        TunnellingMatrix,
        TunnellingState,
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
    shape: _S0Inv,
) -> list[TunnellingState[_S0Inv]]:
    """
    Given the shape of the tunnelling state, get all tunnelling states located at the (0,0) site.

    Parameters
    ----------
    matrix : TunnellingMatrix[_S0Inv]

    Returns
    -------
    TunnellingState[_S0Inv]
    """
    states: list[TunnellingState[_S0Inv]] = []
    for n in range(shape[2]):
        vector = np.zeros(shape)
        vector[0, 0, n] = 1
        states.append({"shape": shape, "vector": vector.flatten()})
    return states


def get_occupation_per_state(
    state: TunnellingState[tuple[_L0Inv, _L1Inv, _L2Inv]]
) -> np.ndarray[tuple[_L2Inv], np.dtype[np.float_]]:
    """
    Get the total occupation probability of each state in a given TunnellingState.

    Parameters
    ----------
    state : TunnellingState[tuple[_L0Inv, _L1Inv, _L2Inv]]

    Returns
    -------
    np.ndarray[tuple[_L2Inv], np.dtype[np.float_]]
    """
    return np.sum(state["vector"].reshape(*state["shape"]), axis=(0, 1))  # type: ignore[no-any-return]


def get_occupation_per_site(
    state: TunnellingState[tuple[_L0Inv, _L1Inv, _L2Inv]]
) -> np.ndarray[tuple[_L0Inv, _L1Inv], np.dtype[np.float_]]:
    """
    Get the total occupation probability of each site in a given TunnellingState.

    Parameters
    ----------
    state : TunnellingState[tuple[_L0Inv, _L1Inv, _L2Inv]]

    Returns
    -------
    np.ndarray[tuple[_L0Inv, _L1Inv], np.dtype[np.float_]]
    """
    return np.sum(state["vector"].reshape(*state["shape"]), axis=(2))  # type: ignore[no-any-return]
