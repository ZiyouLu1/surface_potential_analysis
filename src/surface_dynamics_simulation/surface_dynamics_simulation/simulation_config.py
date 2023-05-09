from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Literal, TypedDict, TypeVar

import numpy as np
import scipy
import scipy.linalg
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

if TYPE_CHECKING:
    from collections.abc import Callable

_LCov = TypeVar("_LCov", bound=int)

_NYInv = TypeVar("_NYInv", bound=int)
_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)


class SimulationConfig(TypedDict, Generic[_LCov]):
    """Dict to store coefficients required for simulation."""

    shape: tuple[int, int]
    """
    The number of repeats of the unit cell.
    """
    # TODO: complex??
    coherent_coefficients: np.ndarray[
        tuple[_LCov, Literal[3], Literal[3]], np.dtype[np.float_]
    ]
    incoherent_coefficients: np.ndarray[tuple[_LCov, _LCov], np.dtype[np.complex_]]


def _get_solver_function(
    config: SimulationConfig[_L0Inv],
) -> Callable[
    [np.float_, np.ndarray[tuple[_NYInv], np.dtype[np.complex_]]],
    np.ndarray[tuple[_NYInv], np.dtype[np.complex_]],
]:
    def fun(
        _t: np.float_, y: np.ndarray[tuple[_NYInv], np.dtype[np.complex_]]
    ) -> np.ndarray[tuple[_NYInv], np.dtype[np.complex_]]:
        """Return dy/dt for the solver given y."""
        stacked = y.reshape(*config["shape"], config["coherent_coefficients"].shape[0])
        stacked += (
            1j
            * config["coherent_coefficients"][np.newaxis, np.newaxis, :, 0, 0]
            * stacked
        )
        return stacked.reshape(-1)

    return fun


def build_tunnelling_matrix(
    coefficients: np.ndarray[tuple[_L0Inv, _L0Inv, Literal[9]], np.dtype[np.float_]],
    shape: tuple[int, int],
) -> np.ndarray[tuple[int, int], np.dtype[np.float_]]:
    """
    Given a list of hopping coefficients build the tunnelling matrix for a grid of the given shape.

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

    the resulting hopping matrix np.ndarray[tuple[int, int], np.dtype[np.float_]] M[i,j]
    gives the total net rate of tunnelling from site i to site j, indexed such that
    M.reshape(shape[0], shape[1],n_states,shape[0], shape[1],n_states)[i0,j0,n0,i1,j1,n1]
    = MS[i0,j0,n0,i1,j1,n1]
    gives the total change in the occupation probability at state [i0,j0,n0] per the probability
    of occupation at site [i1,j1,n1]
    ie d P[i0,j0,n0] / dt = MS[i0,j0,n0,i1,j1,n1] P[i1,j1,n1]
    Note: M[i,i] = -sum_j M[j,i] i != j, since the

    Parameters
    ----------
    coefficients : np.ndarray[tuple[_L0Inv, _L0Inv, Literal[8]], np.dtype[np.float_]]
    shape : tuple[int, int]

    Returns
    -------
    np.ndarray[tuple[int, int], np.dtype[np.float_]]
    """
    n_states = coefficients.shape[0]
    out = np.zeros((*shape, n_states, *shape, n_states))

    for ix0, ix1, n in np.ndindex((*shape, n_states)):
        for dix0 in [-1, 0, 1]:
            for dix1 in [-1, 0, 1]:
                for m in range(n_states):
                    # Calculate the hopping rate out from state ix0, ix1, n
                    # in to the state jx0, jx1 m
                    jx0 = (ix0 + dix0) % out.shape[0]
                    jx1 = (ix1 + dix1) % out.shape[1]

                    hop_idx = np.ravel_multi_index((dix0, dix1), (3, 3), mode="wrap")
                    rate = coefficients[n, m, hop_idx]
                    # Add the contribution from this rate to the total

                    # negative sign as this is a rate out from site ix0, ix1, n
                    # due to the occupation at ix0, ix1, n
                    out[ix0, ix1, n, ix0, ix1, n] -= rate
                    # positive sign as this is a rate in to site jx0, jx1, m
                    out[jx0, jx1, m, ix0, ix1, n] += rate

    return out.reshape(np.prod(shape) * n_states, np.prod(shape) * n_states)  # type: ignore[no-any-return]


def simulate_tunnelling(  # noqa: D103
    matrix: np.ndarray[tuple[_L0Inv, _L0Inv], np.dtype[np.float_]],
    initial_state: np.ndarray[tuple[_L0Inv], np.dtype[np.float_]],
    times: np.ndarray[tuple[_L1Inv], np.dtype[np.float_]],
) -> np.ndarray[tuple[_L0Inv, _L1Inv], np.dtype[np.float_]]:
    energies, vectors = scipy.linalg.eig(matrix)
    print(energies.shape, vectors.shape)  # noqa: T201

    solved = scipy.linalg.solve(vectors, initial_state)
    print(solved.shape)  # noqa: T201

    constants = solved[:, np.newaxis] * np.exp(
        energies[:, np.newaxis] * times[np.newaxis, :]
    )

    return np.sum(vectors[:, :, np.newaxis] * constants[np.newaxis], axis=1)  # type: ignore[no-any-return]


if __name__ == "__main__":
    config: SimulationConfig[Literal[1]] = {
        "coherent_coefficients": np.array([[[1e-8, 0, 0], [0, 0, 0], [0, 0, 0]]]),
        "incoherent_coefficients": np.array([[0]]),
        "shape": (1, 1),
    }
    y0 = np.zeros(
        (*config["shape"], config["coherent_coefficients"].shape[0]), dtype=complex
    )
    y0[0, 0, 0] = 1

    out = solve_ivp(
        _get_solver_function(config),
        t_span=(0, 1),
        method="RK45",
        y0=y0.flatten(),
    )

    out = solve_ivp(
        lambda _t, _y: 0.000001j,
        t_span=(0, 1),
        method="RK45",
        y0=[0 + 1j],
    )

    fig, ax = plt.subplots()
    (line,) = ax.plot(np.real(out.y[0, :]), np.imag(out.y[0, :]))
    (line,) = ax.twinx().twiny().plot(out.t, np.abs(out.y[0, :]))
    line.set_marker("x")
    fig.show()
    input()
