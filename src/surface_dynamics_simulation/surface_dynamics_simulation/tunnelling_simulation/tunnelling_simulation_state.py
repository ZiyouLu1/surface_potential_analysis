from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypedDict, TypeVar

from surface_potential_analysis.basis_config.basis_config import BasisConfig

if TYPE_CHECKING:
    import numpy as np

_S0Inv = TypeVar("_S0Inv", bound=tuple[int, int, int])
_N0Inv = TypeVar("_N0Inv", bound=int)
_BC0Inv = TypeVar("_BC0Inv", bound=BasisConfig[Any, Any, Any])


class TunnellingSimulationState(TypedDict, Generic[_N0Inv, _S0Inv]):
    """Represents the result of a tunnelling simulation."""

    times: np.ndarray[tuple[_N0Inv], np.dtype[np.float_]]
    vectors: np.ndarray[tuple[int, _N0Inv], np.dtype[np.float_]]
    """
    Vector represented such that vector.reshape(shape,-1)[i,j,n,t]
    gives the relevant occupancy of the i,jth site in the nth band
    at a time times[t]
    """
    shape: _S0Inv


class TunnellingSimulationState2(TypedDict, Generic[_N0Inv, _BC0Inv]):
    """Represents the result of a tunnelling simulation."""

    basis: _BC0Inv

    times: np.ndarray[tuple[_N0Inv], np.dtype[np.float_]]
    vectors: np.ndarray[tuple[int, _N0Inv], np.dtype[np.float_]]
    """
    Vector represented such that vector[i,t]
    gives the relevant occupancy of the ith site in the basis
    at a time times[t]
    """
