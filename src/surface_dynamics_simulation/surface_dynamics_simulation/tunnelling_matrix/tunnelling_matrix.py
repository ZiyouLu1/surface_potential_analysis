from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypedDict, TypeVar

from surface_potential_analysis.basis_config.basis_config import BasisConfig

if TYPE_CHECKING:
    import numpy as np

_S0Inv = TypeVar("_S0Inv", bound=tuple[int, int, int])


class TunnellingMatrix(TypedDict, Generic[_S0Inv]):
    """
    Represents the tunnelling coefficients on a grid of _S0Inv points, with _L0Inv bands.

    The resulting matrix np.ndarray[tuple[int, int], np.dtype[np.float_]] M[i,j]
    gives the total net rate of tunnelling from site i to site j, indexed such that
    M.reshape(shape[0], shape[1],n_states,shape[0], shape[1],n_states)[i0,j0,n0,i1,j1,n1]
    = MS[i0,j0,n0,i1,j1,n1]
    gives the total change in the occupation probability at state [i0,j0,n0] per the probability
    of occupation at site [i1,j1,n1]
    ie d P[i0,j0,n0] / dt = MS[i0,j0,n0,i1,j1,n1] P[i1,j1,n1]
    Note: M[i,i] = -sum_j M[j,i] i != j, as probability is conserved
    """

    array: np.ndarray[tuple[int, int], np.dtype[np.float_]]
    """
    Array represented such that array.reshape(shape, shape)[i0,j0,n0,i1,j1,n1]
    gives the total change in the occupation probability at state [i0,j0,n0] per the probability
    of occupation at site [i1,j1,n1]
    """
    shape: _S0Inv


class TunnellingState(TypedDict, Generic[_S0Inv]):
    """Represents the state of a tunnelling simulation at a given time."""

    vector: np.ndarray[tuple[int], np.dtype[np.float_]]
    """
    Vector represented such that vector.reshape(shape)[i,j,n]
    gives the relevant occupancy of the i,jth site in the nth band
    """
    shape: _S0Inv


_BC0Inv = TypeVar("_BC0Inv", bound=BasisConfig[Any, Any, Any])


class TunnellingMatrix2(TypedDict, Generic[_BC0Inv]):
    """
    Represents the tunnelling coefficients between states on a grid given by basis.

    The resulting matrix np.ndarray[tuple[int, int], np.dtype[np.float_]] M[i,j]
    gives the total net rate of tunnelling from site i to site j, indexed such that
    M[i,j]
    gives the total change in the occupation probability at state [i] per the probability
    of occupation at site [j]
    ie d P[i] / dt = MS[j] P[j]
    Note: M[i,i] = -sum_j M[j,i] i != j, as probability is conserved
    """

    array: np.ndarray[tuple[int, int], np.dtype[np.float_]]
    """
    Array represented such that array.reshape(shape, shape)[i0,j0,n0,i1,j1,n1]
    gives the total change in the occupation probability at state [i0,j0,n0] per the probability
    of occupation at site [i1,j1,n1]
    """
    basis: _BC0Inv
    """
    Basis used to represent the states in Tunnelling matrix
    NOTE: the delta_x for this basis spans the region of the whole simulation
    """


class TunnellingState2(TypedDict, Generic[_BC0Inv]):
    """Represents the state of a tunnelling simulation at a given time."""

    vector: np.ndarray[tuple[int], np.dtype[np.float_]]
    """
    Vector represented such that vector.reshape(shape)[i,j,n]
    gives the relevant occupancy of the i,jth site in the nth band
    """
    basis: _BC0Inv
    """
    Basis used to represent the states in Tunnelling matrix
    NOTE: the delta_x for this basis spans the region of the whole simulation
    """
