from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypedDict, TypeVar

if TYPE_CHECKING:
    import numpy as np

_S0Inv = TypeVar("_S0Inv", bound=tuple[int, int, int])
_N0Inv = TypeVar("_N0Inv", bound=int)


class TunnellingVector(TypedDict, Generic[_S0Inv]):
    """Represents the state of a tunnelling simulation at a given time."""

    vector: np.ndarray[tuple[int], np.dtype[np.float_]]
    """
    Vector represented such that vector.reshape(shape)[i,j,n]
    gives the relevant occupancy of the i,jth site in the nth band
    """
    shape: _S0Inv


class TunnellingSimulationState(TypedDict, Generic[_N0Inv, _S0Inv]):
    """Represents the result of a tunnelling simulation."""

    times: np.ndarray[tuple[_N0Inv], np.dtype[np.float_]]
    vectors: np.ndarray[tuple[int, _N0Inv], np.dtype[np.float_]]
    """
    Vector represented such that vector.reshape(shape)[i,j,n]
    gives the relevant occupancy of the i,jth site in the nth band
    """
    shape: _S0Inv
